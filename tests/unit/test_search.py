import asyncio
import unittest
from unittest.mock import AsyncMock

from tenacity import AsyncRetrying, wait_none

from deep_research_agent.services.search.base import (
    BaseSearchProvider,
    SearchErrorType,
    SearchProviderError,
    SearchQuery,
    SearchResponse,
    SearchResultItem,
)
from deep_research_agent.services.search.service import SearchService


class TestSearchService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Use AsyncMock for async methods on the provider
        self.mock_provider = AsyncMock(spec=BaseSearchProvider)
        self.service = SearchService(provider=self.mock_provider)
        # Keep retry-based tests fast by disabling the exponential backoff sleep.
        # tenacity attaches the Retrying object as `retry` at runtime (not
        # visible to the type checker), so fetch it via getattr.
        retrying = SearchService._search_with_retries.retry  # pyright: ignore[reportFunctionMemberAccess]
        if isinstance(retrying, AsyncRetrying):
            retrying.wait = wait_none()

    def assert_error_response(
        self,
        response: SearchResponse,
        error_type: SearchErrorType,
    ):
        self.assertFalse(response.success)
        self.assertIs(response.error_type, error_type)
        self.assertEqual(response.results, [])
        self.assertIsNotNone(response.search_id)

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

    async def test_empty_query_returns_validation_error_response(self):
        response = await self.service.execute_search("")

        self.assert_error_response(response, SearchErrorType.validation)
        self.assertEqual(response.error, "Query must be a non-empty string")
        self.assertEqual(response.query, "")

    async def test_max_results_out_of_range_returns_validation_error_response(self):
        for bad in (0, 11):
            with self.subTest(max_results=bad):
                response = await self.service.execute_search("query", max_results=bad)

                self.assert_error_response(response, SearchErrorType.validation)
                self.assertEqual(response.error, "max_results must be between 1 and 10")

    async def test_non_string_query_returns_validation_error_response(self):
        response = await self.service.execute_search(123)  # type: ignore[arg-type]

        self.assert_error_response(response, SearchErrorType.validation)
        self.assertEqual(response.error, "Query must be a non-empty string")

    async def test_whitespace_only_query_returns_validation_error_response(self):
        for bad in ("   ", "\t ", " \n "):
            with self.subTest(query=bad):
                response = await self.service.execute_search(bad)

                self.assert_error_response(response, SearchErrorType.validation)
                self.assertEqual(response.error, "Query must be a non-empty string")

    async def test_query_is_stripped(self):
        self.mock_provider.search.return_value = []

        response = await self.service.execute_search("  ai agents  ")

        self.assertEqual(response.query, "ai agents")
        actual_query: SearchQuery = self.mock_provider.search.call_args[0][0]
        self.assertEqual(actual_query.query, "ai agents")

    async def test_auth_error_returns_safe_auth_response(self):
        self.mock_provider.search.side_effect = SearchProviderError(
            SearchErrorType.auth, "InvalidAPIKeyError: 401 sk-leaked-secret-key"
        )

        response = await self.service.execute_search("python web agents")

        self.assert_error_response(response, SearchErrorType.auth)
        self.assertEqual(response.error, "Search provider rejected the API key")
        error = response.error
        assert error is not None
        self.assertNotIn("sk-leaked-secret-key", error)
        self.mock_provider.search.assert_called_once()  # auth is not retried

    async def test_rate_limit_error_returns_rate_limit_response(self):
        self.mock_provider.search.side_effect = SearchProviderError(
            SearchErrorType.rate_limit, "Usage rate limited"
        )

        response = await self.service.execute_search("python web agents")

        self.assert_error_response(response, SearchErrorType.rate_limit)
        self.assertEqual(response.error, "Search provider rate limit exceeded")

    async def test_unexpected_exception_returns_provider_error_response(self):
        self.mock_provider.search.side_effect = RuntimeError(
            "unexpected provider crash"
        )

        response = await self.service.execute_search("python web agents")

        self.assert_error_response(response, SearchErrorType.provider)
        self.assertEqual(response.error, "Search provider failed")

    async def test_network_error_retried_then_returns_network_response(self):
        self.mock_provider.search.side_effect = SearchProviderError(
            SearchErrorType.network, "connect failed"
        )

        response = await self.service.execute_search("python web agents")

        self.assert_error_response(response, SearchErrorType.network)
        self.assertEqual(response.error, "Search provider unreachable; try again later")
        # 1 initial attempt + 2 retries (stop_after_attempt(3))
        self.assertEqual(self.mock_provider.search.call_count, 3)

    async def test_transient_network_error_recovers_after_retry(self):
        results = [
            SearchResultItem(
                url="https://example.com/page1",
                title="Page 1",
                content="Content 1",
            )
        ]
        self.mock_provider.search.side_effect = [
            SearchProviderError(SearchErrorType.network, "connect failed"),
            results,
        ]

        response = await self.service.execute_search("python web agents")

        self.assertTrue(response.success)
        self.assertEqual(response.results, results)
        self.assertEqual(self.mock_provider.search.call_count, 2)
        self.assertIsNone(response.error_type)

    async def test_batch_search_returns_response_per_query_without_cancelling(self):
        self.mock_provider.search.side_effect = self._fail_for_queries(
            failing={"bad-key": SearchErrorType.auth}
        )

        responses = await self.service.execute_batch_search(
            ["good", "bad-key", "", "  padded  "]
        )

        self.assertEqual(len(responses), 4)
        self.assertTrue(responses[0].success)
        self.assert_error_response(responses[1], SearchErrorType.auth)
        self.assert_error_response(responses[2], SearchErrorType.validation)
        self.assertTrue(responses[3].success)
        self.assertEqual(responses[3].query, "padded")

    async def test_batch_search_bounds_concurrency(self):
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def _search(query: SearchQuery) -> list[SearchResultItem]:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            async with lock:
                active -= 1
            return []

        self.mock_provider.search.side_effect = _search
        self.service = SearchService(provider=self.mock_provider, max_concurrency=2)

        responses = await self.service.execute_batch_search(["a", "b", "c", "d", "e"])

        self.assertEqual(len(responses), 5)
        self.assertLessEqual(max_active, 2)

    def _fail_for_queries(self, failing: dict[str, SearchErrorType]):
        async def _search(query: SearchQuery):
            error_type = failing.get(query.query)
            if error_type is not None:
                raise SearchProviderError(error_type, f"{error_type} error")
            return []

        return _search


if __name__ == "__main__":
    unittest.main()
