import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from deep_research_agent.services.search.base import (
    SearchErrorType,
    SearchProviderError,
    SearchQuery,
    SearchResultItem,
)
from deep_research_agent.services.search.exa import ExaSearchProvider


class TestExaSearchProvider(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = ExaSearchProvider(api_key="test-key")

    async def test_http_error_classification(self):
        for status, expected in (
            (401, SearchErrorType.auth),
            (429, SearchErrorType.rate_limit),
            (500, SearchErrorType.network),
            (503, SearchErrorType.network),
            (400, SearchErrorType.provider),
            (404, SearchErrorType.provider),
        ):
            with (
                self.subTest(status=status),
                patch.object(
                    self.provider._client,
                    "search",
                    new=AsyncMock(
                        side_effect=ValueError(
                            f"Request failed with status code {status}: oops"
                        )
                    ),
                ),
                self.assertRaises(SearchProviderError) as ctx,
            ):
                await self.provider.search(SearchQuery(query="agents"))

            self.assertIs(ctx.exception.error_type, expected)

    async def test_value_error_without_status_is_provider_error(self):
        with (
            patch.object(
                self.provider._client,
                "search",
                new=AsyncMock(side_effect=ValueError("boom")),
            ),
            self.assertRaises(SearchProviderError) as ctx,
        ):
            await self.provider.search(SearchQuery(query="agents"))

        self.assertIs(ctx.exception.error_type, SearchErrorType.provider)

    async def test_network_error_is_network_error(self):
        with (
            patch.object(
                self.provider._client,
                "search",
                new=AsyncMock(side_effect=httpx.ConnectError("unreachable")),
            ),
            self.assertRaises(SearchProviderError) as ctx,
        ):
            await self.provider.search(SearchQuery(query="agents"))

        self.assertIs(ctx.exception.error_type, SearchErrorType.network)

    async def test_malformed_results_raise_provider_error(self):
        result = SimpleNamespace(
            url="https://example.com",
            title="Example",
            text=None,
            summary=None,
            highlights=None,
            score=2.0,
        )
        response = SimpleNamespace(results=[result])

        with (
            patch.object(
                self.provider._client,
                "search",
                new=AsyncMock(return_value=response),
            ),
            self.assertRaises(SearchProviderError) as ctx,
        ):
            await self.provider.search(SearchQuery(query="agents"))

        self.assertIs(ctx.exception.error_type, SearchErrorType.provider)

    async def test_valid_results_are_mapped(self):
        result = SimpleNamespace(
            url="https://example.com",
            title="Example",
            text="hello world",
            summary=None,
            highlights=None,
            score=0.8,
        )
        response = SimpleNamespace(results=[result])

        with patch.object(
            self.provider._client, "search", new=AsyncMock(return_value=response)
        ):
            results = await self.provider.search(SearchQuery(query="agents"))

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], SearchResultItem)
        self.assertEqual(results[0].url, "https://example.com")
        self.assertEqual(results[0].content, "hello world")
        self.assertEqual(results[0].score, 0.8)


if __name__ == "__main__":
    unittest.main()
