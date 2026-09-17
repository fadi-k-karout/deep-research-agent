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

        # Extract the list of result dicts
        raw_results = response.get("results", [])

        items = [
            SearchResultItem(
                url=item["url"],
                title=item.get("title", ""),
                content=item.get("content", ""),
                score=item.get("score") or 0.0,
            )
            for item in raw_results
        ]

        # Map explicitly to SearchResultItem schema
        return items
