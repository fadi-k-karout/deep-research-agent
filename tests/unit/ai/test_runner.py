import asyncio
import unittest
from unittest.mock import AsyncMock

from deep_research_agent.ai.agent.runner import (
    AgentRunner,
    ExtractedFindings,
    FindingDraft,
    PlannedQueries,
)
from deep_research_agent.ai.llm.base import (
    BaseLLMProvider,
    LLMRequest,
    LLMResponse,
)
from deep_research_agent.services.search.base import (
    SearchErrorType,
    SearchResponse,
    SearchResultItem,
)
from deep_research_agent.services.search.service import SearchService


def _llm_response(content) -> LLMResponse:
    return LLMResponse(content=content, raw_response="{}", model_name="test-model")


def _search_response(query: str, results: list[SearchResultItem]) -> SearchResponse:
    return SearchResponse(search_id="1", query=query, results=results)


def _result(title: str = "Title", url: str = "https://example.com") -> SearchResultItem:
    return SearchResultItem(title=title, url=url, content="Some content.")


class TestAgentRunner(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.llm = AsyncMock(spec=BaseLLMProvider)
        self.search = AsyncMock(spec=SearchService)
        self.runner = AgentRunner(search_service=self.search, llm=self.llm)

    async def test_full_loop_plans_searches_extracts_and_synthesizes(self):
        self.llm.generate.side_effect = [
            _llm_response(PlannedQueries(queries=["what is x"], is_complete=False)),
            _llm_response(
                ExtractedFindings(findings=[FindingDraft(title="T", content="C")])
            ),
            _llm_response(PlannedQueries(queries=[], is_complete=True)),
            _llm_response("# Research Report"),
        ]
        self.search.execute_batch_search.return_value = [
            _search_response("what is x", [_result(url="https://example.com/x")])
        ]

        state = await self.runner.run("Research x")

        self.assertEqual(state.current_iteration, 2)
        self.assertTrue(state.is_complete)
        self.assertEqual(len(state.findings), 1)
        finding = state.findings[0]
        self.assertEqual(finding.title, "T")
        self.assertEqual(finding.content, "C")
        self.assertEqual(finding.source_url, "https://example.com/x")
        self.assertEqual(finding.query_used, "what is x")
        self.assertEqual(state.synthesized_report, "# Research Report")
        self.search.execute_batch_search.assert_awaited_once_with(["what is x"])

    async def test_failed_search_response_is_skipped(self):
        self.llm.generate.side_effect = [
            _llm_response(PlannedQueries(queries=["q"], is_complete=False)),
            _llm_response(PlannedQueries(queries=[], is_complete=True)),
            _llm_response("# Research Report"),
        ]
        self.search.execute_batch_search.return_value = [
            SearchResponse(
                search_id="1",
                query="q",
                results=[],
                success=False,
                error="boom",
                error_type=SearchErrorType.provider,
            )
        ]

        state = await self.runner.run("Research x")

        self.assertEqual(state.findings, [])
        self.assertEqual(state.failed_searches, 1)
        self.assertEqual(state.search_failures, ["q: boom"])
        # No extraction or synthesis calls after the failed response.
        self.assertEqual(self.llm.generate.await_count, 2)
        report = state.synthesized_report
        assert report is not None
        self.assertIn("No findings were gathered", report)
        self.assertIn("q: boom", report)

    async def test_respects_max_iterations(self):
        self.runner = AgentRunner(
            search_service=self.search, llm=self.llm, max_iterations=3
        )

        async def _generate(request: LLMRequest):
            prompt = request.prompt
            if "Write the final research report" in prompt:
                return _llm_response("# Research Report")
            if "URL: " in prompt:
                return _llm_response(
                    ExtractedFindings(findings=[FindingDraft(title="T", content="C")])
                )
            return _llm_response(PlannedQueries(queries=["q"], is_complete=False))

        self.llm.generate.side_effect = _generate
        self.search.execute_batch_search.return_value = [
            _search_response("q", [_result(url="https://example.com/1")])
        ]

        state = await self.runner.run("Research x")

        self.assertEqual(state.current_iteration, 3)
        self.assertEqual(state.stalled_iterations, 0)
        self.assertEqual(state.synthesized_report, "# Research Report")

    async def test_no_queries_planned_stops_immediately(self):
        self.llm.generate.side_effect = [
            _llm_response(PlannedQueries(queries=[], is_complete=False)),
            _llm_response("# Research Report"),
        ]

        state = await self.runner.run("Research x")

        self.assertEqual(state.current_iteration, 1)
        self.assertTrue(state.is_complete)
        self.assertEqual(state.findings, [])
        self.assertEqual(self.llm.generate.await_count, 1)
        report = state.synthesized_report
        assert report is not None
        self.assertIn("No findings were gathered", report)
        self.assertIn("planner ended the session", report)

    async def test_planning_failure_stops_loop(self):
        def _generate(request: LLMRequest):
            if "Write the final research report" in request.prompt:
                return _llm_response("# Research Report")
            raise ValueError("bad json")

        self.llm.generate.side_effect = _generate

        state = await self.runner.run("Research x")

        self.assertEqual(state.current_iteration, 1)
        self.assertTrue(state.is_complete)
        self.assertEqual(state.findings, [])
        self.assertEqual(self.llm.generate.await_count, 2)
        report = state.synthesized_report
        assert report is not None
        self.assertIn("No findings were gathered", report)

    async def test_synthesis_failure_returns_fallback_report(self):
        self.llm.generate.side_effect = [
            _llm_response(PlannedQueries(queries=["q"], is_complete=False)),
            _llm_response(
                ExtractedFindings(findings=[FindingDraft(title="T", content="C")])
            ),
            _llm_response(PlannedQueries(queries=[], is_complete=True)),
            ValueError("bad json"),
        ]
        self.search.execute_batch_search.return_value = [
            _search_response("q", [_result(url="https://example.com/1")])
        ]

        state = await self.runner.run("Research x")

        report = state.synthesized_report
        assert report is not None
        self.assertIn("# Research Report", report)
        self.assertIn("Research x", report)
        self.assertIn("T", report)

    async def test_extraction_failure_skips_result_but_continues(self):
        def _generate(request: LLMRequest):
            prompt = request.prompt
            if "Write the final research report" in prompt:
                return _llm_response("# Research Report")
            if "URL: " in prompt and "example.com/bad" in prompt:
                raise ValueError("bad json")
            if "URL: " in prompt and "example.com/good" in prompt:
                return _llm_response(
                    ExtractedFindings(findings=[FindingDraft(title="T", content="C")])
                )
            if "None yet." in prompt:
                return _llm_response(PlannedQueries(queries=["q"], is_complete=False))
            return _llm_response(PlannedQueries(queries=[], is_complete=True))

        self.llm.generate.side_effect = _generate
        self.search.execute_batch_search.return_value = [
            _search_response(
                "q",
                [
                    _result(url="https://example.com/bad"),
                    _result(url="https://example.com/good"),
                ],
            )
        ]

        state = await self.runner.run("Research x")

        self.assertEqual(len(state.findings), 1)
        self.assertEqual(state.findings[0].source_url, "https://example.com/good")

    async def test_extraction_runs_results_concurrently(self):
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def _generate(request: LLMRequest):
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            async with lock:
                active -= 1

            prompt = request.prompt
            if "Write the final research report" in prompt:
                return _llm_response("# Research Report")
            if "None yet." in prompt:
                return _llm_response(PlannedQueries(queries=["q"], is_complete=False))
            if "URL: " in prompt:
                return _llm_response(
                    ExtractedFindings(findings=[FindingDraft(title="T", content="C")])
                )
            return _llm_response(PlannedQueries(queries=[], is_complete=True))

        self.llm.generate.side_effect = _generate
        self.search.execute_batch_search.return_value = [
            _search_response(
                "q",
                [
                    _result(url="https://example.com/1"),
                    _result(url="https://example.com/2"),
                    _result(url="https://example.com/3"),
                ],
            )
        ]

        state = await self.runner.run("Research x")

        self.assertGreaterEqual(max_active, 2)
        self.assertEqual(
            [f.source_url for f in state.findings],
            [
                "https://example.com/1",
                "https://example.com/2",
                "https://example.com/3",
            ],
        )

    async def test_all_searches_failed_report_explains_failure(self):
        self.llm.generate.side_effect = [
            _llm_response(PlannedQueries(queries=["q"], is_complete=False)),
            _llm_response(PlannedQueries(queries=[], is_complete=True)),
        ]
        self.search.execute_batch_search.return_value = [
            SearchResponse(
                search_id="1",
                query="q",
                results=[],
                success=False,
                error="Search provider failed",
                error_type=SearchErrorType.provider,
            )
        ]

        state = await self.runner.run("Research x")

        self.assertEqual(state.failed_searches, 1)
        self.assertEqual(state.search_failures, ["q: Search provider failed"])
        self.assertEqual(state.stalled_iterations, 1)
        # The LLM never fabricates a report from no findings.
        self.assertEqual(self.llm.generate.await_count, 2)
        report = state.synthesized_report
        assert report is not None
        self.assertIn("No findings were gathered", report)
        self.assertIn("q: Search provider failed", report)

    async def test_stops_after_consecutive_stalled_iterations(self):
        self.runner = AgentRunner(
            search_service=self.search, llm=self.llm, max_iterations=5
        )

        def _generate(request: LLMRequest):
            return _llm_response(PlannedQueries(queries=["q"], is_complete=False))

        self.llm.generate.side_effect = _generate
        self.search.execute_batch_search.return_value = [_search_response("q", [])]

        state = await self.runner.run("Research x")

        self.assertEqual(state.current_iteration, 2)
        self.assertEqual(state.stalled_iterations, 2)
        self.assertEqual(state.failed_searches, 2)
        self.assertTrue(state.is_complete)
        report = state.synthesized_report
        assert report is not None
        self.assertIn("q: no results returned", report)

    async def test_report_appends_research_notes_when_searches_failed(self):
        bad = SearchResponse(
            search_id="1",
            query="bad",
            results=[],
            success=False,
            error="Search provider failed",
            error_type=SearchErrorType.provider,
        )
        good = _search_response("good", [_result(url="https://example.com/good")])
        self.llm.generate.side_effect = [
            _llm_response(PlannedQueries(queries=["bad", "good"], is_complete=False)),
            _llm_response(
                ExtractedFindings(findings=[FindingDraft(title="T", content="C")])
            ),
            _llm_response(PlannedQueries(queries=[], is_complete=True)),
            _llm_response("## Report Body"),
        ]
        self.search.execute_batch_search.return_value = [bad, good]

        state = await self.runner.run("Research x")

        self.assertEqual(state.failed_searches, 1)
        self.assertEqual(len(state.findings), 1)
        report = state.synthesized_report

        assert report is not None
        self.assertIn("## Report Body", report)
        self.assertIn("Research Notes", report)
        self.assertIn("bad: Search provider failed", report)
        synthesis_prompt = self.llm.generate.call_args_list[3].args[0].prompt
        self.assertIn("bad: Search provider failed", synthesis_prompt)


if __name__ == "__main__":
    unittest.main()
