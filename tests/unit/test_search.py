import unittest
from unittest.mock import AsyncMock

from deep_research_agent.services.search.base import (
    BaseSearchProvider,
    SearchQuery,
    SearchResultItem,
)
from deep_research_agent.services.search.service import SearchService


class TestSearchService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Use AsyncMock for async methods on the provider
        self.mock_provider = AsyncMock(spec=BaseSearchProvider)
        self.service = SearchService(provider=self.mock_provider)

    async def test_execute_search_returns_provider_results(self):
        results = [
            SearchResultItem(
                url="https://example.com/page1",
                title="Page 1",
                content="Content 1",
            ),
            SearchResultItem(
                url="https://example.com/page2/",
                title="Page 2",
                content="Content 2",
            ),
        ]

        self.mock_provider.search.return_value = results

        response = await self.service.execute_search("python web agents")

        self.assertEqual(response.results, results)
        self.assertEqual(response.query, "python web agents")
        self.assertIsNotNone(response.search_id)

    async def test_empty_results_from_provider(self):
        """Verifies service handles empty provider responses gracefully."""
        query_str = "nonexistent term"

        self.mock_provider.search.return_value = []

        response = await self.service.execute_search(query_str)
        self.assertEqual(len(response.results), 0)

        # Assert that provider was called with a SearchQuery instance containing query_str
        self.mock_provider.search.assert_called_once()

        # Verify the single SearchQuery argument directly
        actual_query: SearchQuery = self.mock_provider.search.call_args[0][0]
        self.assertIsInstance(actual_query, SearchQuery)
        self.assertEqual(actual_query.query, query_str)
        self.assertEqual(actual_query.max_results, 5)

    async def test_empty_query_raises(self):
        with self.assertRaises(ValueError):
            await self.service.execute_search("")

    async def test_max_results_out_of_range_raises_value_error(self):
        for bad in (0, 11):
            with self.assertRaises(ValueError):
                await self.service.execute_search("query", max_results=bad)

    async def test_non_string_query_raises_value_error(self):
        with self.assertRaises(ValueError):
            await self.service.execute_search(123)  # type: ignore[arg-type]

    async def test_whitespace_only_query_raises_value_error(self):
        for bad in ("   ", "\t ", " \n "):
            with self.assertRaises(ValueError):
                await self.service.execute_search(bad)

    async def test_query_is_stripped(self):
        self.mock_provider.search.return_value = []

        response = await self.service.execute_search("  ai agents  ")

        self.assertEqual(response.query, "ai agents")
        actual_query: SearchQuery = self.mock_provider.search.call_args[0][0]
        self.assertEqual(actual_query.query, "ai agents")


if __name__ == "__main__":
    unittest.main()
