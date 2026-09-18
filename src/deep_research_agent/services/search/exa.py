import httpx
from exa_py import AsyncExa
from exa_py.api import Result

from deep_research_agent.config import exa_api_key

from .base import (
    BaseSearchProvider,
    SearchErrorType,
    SearchProviderError,
    SearchQuery,
    SearchResultItem,
)


class ExaSearchProvider(BaseSearchProvider):
    def __init__(self, api_key: str | None = None, content_max_chars: int = 10_000):
        self._api_key = api_key or exa_api_key()
        self._content_max_chars = content_max_chars
        self._client = AsyncExa(api_key=self._api_key)

    @staticmethod
    def _extract_content(result: Result) -> str:
        return result.text or result.summary or " ".join(result.highlights or []) or ""

    async def search(self, query: SearchQuery) -> list[SearchResultItem]:
        try:
            response = await self._client.search(
                query=query.query,
                num_results=query.max_results,
                contents={"text": {"max_characters": self._content_max_chars}},
            )
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            raise SearchProviderError(SearchErrorType.network, str(exc)) from exc
        except Exception as exc:
            raise SearchProviderError(SearchErrorType.provider, str(exc)) from exc

        items = [
            SearchResultItem(
                url=result.url,
                title=result.title or "",
                content=ExaSearchProvider._extract_content(result),
                score=result.score or 0.0,
            )
            for result in response.results
        ]

        return items
