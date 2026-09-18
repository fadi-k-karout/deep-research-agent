import asyncio
import logging
from uuid import uuid4

from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from deep_research_agent.services.search.base import (
    BaseSearchProvider,
    SearchErrorType,
    SearchProviderError,
    SearchQuery,
    SearchResponse,
    SearchResultItem,
)

logger = logging.getLogger(__name__)

_SAFE_ERROR_MESSAGES = {
    SearchErrorType.auth: "Search provider rejected the API key",
    SearchErrorType.rate_limit: "Search provider rate limit exceeded",
    SearchErrorType.network: "Search provider unreachable; try again later",
    SearchErrorType.provider: "Search provider failed",
}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, SearchProviderError):
        return exc.error_type is SearchErrorType.network
    return isinstance(exc, (TimeoutError, ConnectionError))


class SearchService:
    def __init__(self, provider: BaseSearchProvider, max_concurrency: int = 5):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._provider = provider
        self._max_concurrency = max_concurrency

    async def execute_search(
        self, query_str: str, max_results: int = 5
    ) -> SearchResponse:
        """
        Execute a search query and always return a SearchResponse.

        This method never raises: validation failures, provider errors, and
        network failures (even after retries are exhausted) are all reported as
        ``success=False`` responses carrying a canonical ``error_type``.

        Args:
            query_str: The search query string.
            max_results: The maximum number of results to return. Defaults to 5.

        Returns:
            SearchResponse: The search results, tagged with a per-search id, or
                an error response when the search could not be completed.
        """
        if not isinstance(query_str, str) or not query_str.strip():
            return self._make_error_response(
                str(query_str),
                SearchErrorType.validation,
                "Query must be a non-empty string",
            )

        query_str = query_str.strip()

        try:
            query = SearchQuery(query=query_str, max_results=max_results)
        except ValidationError:
            return self._make_error_response(
                query_str,
                SearchErrorType.validation,
                "max_results must be between 1 and 10",
            )

        try:
            results = await self._search_with_retries(query)
        except SearchProviderError as exc:
            logger.error("search failed for query=%r: %s", query_str, exc.error_type)
            logger.debug("search failed for query=%r", query_str, exc_info=True)
            message = _SAFE_ERROR_MESSAGES.get(
                exc.error_type, _SAFE_ERROR_MESSAGES[SearchErrorType.provider]
            )
            return self._make_error_response(query_str, exc.error_type, message)
        except Exception:
            logger.error("search failed for query=%r: unexpected error", query_str)
            logger.debug("search failed for query=%r", query_str, exc_info=True)
            return self._make_error_response(
                query_str,
                SearchErrorType.provider,
                _SAFE_ERROR_MESSAGES[SearchErrorType.provider],
            )

        return SearchResponse(
            search_id=str(uuid4()),
            query=query_str,
            results=results,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,  # Surface the original exception so it can be mapped to a response
    )
    async def _search_with_retries(self, query: SearchQuery) -> list[SearchResultItem]:
        return await self._provider.search(query)

    def _make_error_response(
        self,
        query_str: str,
        error_type: SearchErrorType,
        error: str,
    ) -> SearchResponse:
        return SearchResponse(
            search_id=str(uuid4()),
            query=query_str,
            results=[],
            success=False,
            error=error,
            error_type=error_type,
        )

    async def execute_batch_search(self, queries: list[str]) -> list[SearchResponse]:
        """Execute every query and return one SearchResponse per query.

        All queries are run concurrently, but at most ``max_concurrency``
        searches are in flight at once (see :meth:`__init__`). Each query
        keeps its existing retry behavior.

        Args:
            queries: The search query strings.

        Returns:
            list[SearchResponse]: One response per query, in the same order.
        """
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _run_bounded(query: str) -> SearchResponse:
            async with semaphore:
                return await self.execute_search(query)

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(_run_bounded(q)) for q in queries]
        return [task.result() for task in tasks]
