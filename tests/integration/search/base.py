import os
import unittest
from abc import ABC, abstractmethod

from deep_research_agent.config import load_environment
from deep_research_agent.services.search.base import (
    BaseSearchProvider,
    SearchQuery,
    SearchResultItem,
)


class BaseProviderIntegrationTest(unittest.IsolatedAsyncioTestCase, ABC):
    api_key_env: str | None = None

    @abstractmethod
    def _create_provider(self, api_key: str) -> BaseSearchProvider:
        """Instantiate the provider under test with the given API key."""

    async def asyncSetUp(self):
        load_environment()
        if not self.api_key_env:
            raise unittest.SkipTest("api_key_env not configured on test class")
        self.api_key = os.getenv(self.api_key_env)
        if not self.api_key:
            raise unittest.SkipTest(f"{self.api_key_env} environment variable not set")
        self.provider = self._create_provider(self.api_key)

    async def test_live_search_returns_valid_schema(self):
        query = SearchQuery(query="latest advancements in AI agents", max_results=3)
        results = await self.provider.search(query)

        # Assert response shape and Pydantic validation
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # Verify result item schema
        first_result = results[0]
        self.assertIsInstance(first_result, SearchResultItem)
        self.assertTrue(first_result.url.startswith("http"))
        self.assertTrue(len(first_result.title) > 0)
