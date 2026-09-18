import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from deep_research_agent.services.search.base import (
    SearchErrorType,
    SearchProviderError,
    SearchQuery,
    SearchResultItem,
)
from deep_research_agent.services.search.tavily import TavilySearchProvider


class TestTavilySearchProvider(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = TavilySearchProvider(api_key="test-key")

    async def test_invalid_results_raises_provider_error(self):
        for payload in ({"results": {}}, {"results": "oops"}):
            with (
                self.subTest(payload=payload),
                patch.object(
                    self.provider._client,
                    "search",
                    new=AsyncMock(return_value=payload),
                ),
                self.assertRaises(SearchProviderError) as ctx,
            ):
                await self.provider.search(SearchQuery(query="agents"))

            self.assertIs(ctx.exception.error_type, SearchErrorType.provider)

    async def test_empty_results_allowed(self):
        with patch.object(
            self.provider._client,
            "search",
            new=AsyncMock(return_value={"results": []}),
        ):
            results = await self.provider.search(SearchQuery(query="agents"))

        self.assertEqual(results, [])

    async def test_valid_results_are_mapped_and_extra_keys_ignored(self):
        payload = {
            "results": [
                {
                    "url": "https://example.com/page1",
                    "title": "Page 1",
                    "content": "Content 1",
                    "score": 0.8,
                    "published_date": "2026-01-01",
                }
            ]
        }

        with patch.object(
            self.provider._client, "search", new=AsyncMock(return_value=payload)
        ):
            results = await self.provider.search(SearchQuery(query="agents"))

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], SearchResultItem)
        self.assertEqual(results[0].url, "https://example.com/page1")
        self.assertEqual(results[0].score, 0.8)

    async def test_missing_required_field_raises_validation_error(self):
        payload = {"results": [{"url": "https://example.com", "title": "No content"}]}

        with (
            patch.object(
                self.provider._client, "search", new=AsyncMock(return_value=payload)
            ),
            self.assertRaises(ValidationError),
        ):
            await self.provider.search(SearchQuery(query="agents"))


if __name__ == "__main__":
    unittest.main()
