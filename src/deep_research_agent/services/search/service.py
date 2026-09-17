from uuid import uuid4

from pydantic import ValidationError

from deep_research_agent.services.search.base import (
    BaseSearchProvider,
    SearchQuery,
    SearchResponse,
)


class SearchService:
    def __init__(self, provider: BaseSearchProvider):
        self._provider = provider

    async def execute_search(
        self, query_str: str, max_results: int = 5
    ) -> SearchResponse:
        """
        Executes a search query using the configured search provider.

        Args:
            query_str: The search query string.
            max_results: The maximum number of results to return. Defaults to 5.

        Returns:
            SearchResponse: The search results, tagged with a per-search id.

        Raises:
            ValueError: If the query string is empty or max_results is outside
                the supported range (1-10).
        """
        if not query_str:
            raise ValueError("Query cannot be empty")

        try:
            query = SearchQuery(query=query_str, max_results=max_results)
        except ValidationError as exc:
            if any("query" in error["loc"] for error in exc.errors()):
                raise ValueError("Query must be a non-empty string") from exc
            raise ValueError("max_results must be between 1 and 10") from exc
        results = await self._provider.search(query)

        return SearchResponse(
            search_id=str(uuid4()),
            query=query_str,
            results=results,
        )
