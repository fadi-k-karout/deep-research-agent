import httpx
from pydantic import ValidationError
from tavily import (
    AsyncTavilyClient,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)
from tavily.errors import ForbiddenError
from tavily.errors import TimeoutError as TavilyTimeoutError

from deep_research_agent.config import tavily_api_key

from .base import (
    BaseSearchProvider,
    SearchErrorType,
    SearchProviderError,
    SearchQuery,
    SearchResultItem,
)


class TavilySearchProvider(BaseSearchProvider):
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or tavily_api_key()
        self._client = AsyncTavilyClient(api_key=self._api_key)

    async def search(self, query: SearchQuery) -> list[SearchResultItem]:
        try:
            response = await self._client.search(
                query=query.query, max_results=query.max_results
            )
        except (MissingAPIKeyError, InvalidAPIKeyError, ForbiddenError) as exc:
            raise SearchProviderError(SearchErrorType.auth, str(exc)) from exc
        except UsageLimitExceededError as exc:
            raise SearchProviderError(SearchErrorType.rate_limit, str(exc)) from exc
        except (
            TavilyTimeoutError,
            httpx.TimeoutException,
            httpx.ConnectError,
        ) as exc:
            raise SearchProviderError(SearchErrorType.network, str(exc)) from exc
        except Exception as exc:
            raise SearchProviderError(SearchErrorType.provider, str(exc)) from exc

        raw_results = response.get("results", [])
        if not isinstance(raw_results, list):
            raise SearchProviderError(
                SearchErrorType.provider,
                "Unexpected Tavily response: 'results' must be a list, "
                f"got {type(raw_results).__name__}",
            )

        try:
            return [SearchResultItem.model_validate(item) for item in raw_results]
        except ValidationError as exc:
            raise SearchProviderError(SearchErrorType.provider, str(exc)) from exc
