from tavily import AsyncTavilyClient

from deep_research_agent.config import tavily_api_key

from .base import BaseSearchProvider, SearchQuery, SearchResultItem


class TavilySearchProvider(BaseSearchProvider):
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or tavily_api_key()
        self._client = AsyncTavilyClient(api_key=self._api_key)

    async def search(self, query: SearchQuery) -> list[SearchResultItem]:
        response = await self._client.search(
            query=query.query, max_results=query.max_results
        )

        raw_results = response.get("results", [])
        if not isinstance(raw_results, list):
            raise TypeError(
                "Unexpected Tavily response: 'results' must be a list, "
                f"got {type(raw_results).__name__}"
            )

        return [SearchResultItem.model_validate(item) for item in raw_results]
