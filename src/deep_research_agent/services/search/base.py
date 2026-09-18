from abc import ABC, abstractmethod
from enum import StrEnum

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    query: str = Field(..., description="The search query string")
    max_results: int = Field(default=5, ge=1, le=10)


class SearchResultItem(BaseModel):
    title: str
    url: str
    content: str
    score: float | None = Field(default=0.0, ge=0.0, le=1.0)


class SearchErrorType(StrEnum):
    validation = "validation"
    network = "network"
    auth = "auth"
    rate_limit = "rate_limit"
    provider = "provider"


class SearchProviderError(Exception):
    """Raised by providers after translating an SDK/network failure.

    Carries a canonical ``error_type`` so the service can classify failures
    without knowing anything about a specific SDK.
    """

    def __init__(self, error_type: SearchErrorType, detail: str):
        super().__init__(detail)
        self.error_type = error_type
        self.detail = detail


class SearchResponse(BaseModel):
    search_id: str
    query: str
    results: list[SearchResultItem]
    success: bool = Field(default=True)
    error: str | None = Field(default=None)
    error_type: SearchErrorType | None = Field(default=None)


class BaseSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResultItem]:
        """
        Perform a search using the given query and return the results.

        Raises:
            SearchProviderError: Provider and network failures are translated
                by the provider into this canonical error type before they
                reach the caller.
        """
