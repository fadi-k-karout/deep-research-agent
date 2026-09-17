from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    query: str = Field(..., description="The search query string")
    max_results: int = Field(default=5, ge=1, le=10)


class SearchResultItem(BaseModel):
    title: str
    url: str
    content: str
    score: float | None = Field(default=0.0, ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    search_id: str
    query: str
    results: list[SearchResultItem]


class BaseSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResultItem]:
        """
        Perform a search using the given query and return the results.

        Raises:
            ValueError: If the provider's API key is missing (raised at
                construction time).
            Exception: Provider errors (auth failures, rate limits, invalid
                requests) and network errors are propagated as raised by the
                underlying SDK; callers should catch these when retry/fallback
                behaviour is required.
        """
