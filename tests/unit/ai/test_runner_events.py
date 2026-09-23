import asyncio
import unittest
from unittest.mock import AsyncMock

from deep_research_agent.ai.agent.runner import (
    AgentRunner,
    ExtractedFindings,
    FindingDraft,
    PlannedQueries,
)
from deep_research_agent.ai.llm.base import BaseLLMProvider, LLMResponse
from deep_research_agent.events import (
    AgentEvent,
    EventEmitter,
    FindingExtracted,
    IterationCompleted,
    PlanCreated,
    ReportReady,
    RunStarted,
    RunTerminated,
    SearchCompleted,
    SearchResultFound,
    SearchStarted,
    SynthesisStarted,
)
from deep_research_agent.services.search.base import SearchResponse, SearchResultItem
from deep_research_agent.services.search.service import SearchService


def _llm_response(content) -> LLMResponse:
    return LLMResponse(content=content, raw_response="{}", model_name="test-model")


def _search_response(query: str, results: list[SearchResultItem]) -> SearchResponse:
    return SearchResponse(search_id="1", query=query, results=results)


def _result(url: str = "https://example.com") -> SearchResultItem:
    return SearchResultItem(title="Title", url=url, content="Some content.")


class TestAgentRunnerEvents(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.llm = AsyncMock(spec=BaseLLMProvider)
        self.search = AsyncMock(spec=SearchService)
        self.emitter = EventEmitter()
        self.events: list[AgentEvent] = []
        self.emitter.subscribe(self._record)
        self.runner = AgentRunner(
            search_service=self.search,
            llm=self.llm,
            events=self.emitter,
        )

    async def _record(self, event) -> None:
        self.events.append(event)

    async def test_full_run_emits_expected_sequence(self):
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

        self.assertEqual(state.termination_reason, "complete")
        expected = [
            RunStarted,
            PlanCreated,
            SearchStarted,
            SearchCompleted,
            SearchResultFound,
            FindingExtracted,
            IterationCompleted,
            PlanCreated,
            IterationCompleted,
            RunTerminated,
            SynthesisStarted,
            ReportReady,
        ]
        self.assertEqual([type(event) for event in self.events], expected)

    async def test_max_iterations_emits_termination(self):
        self.runner = AgentRunner(
            search_service=self.search,
            llm=self.llm,
            max_iterations=1,
            events=self.emitter,
        )

        async def _generate(request):
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

        await self.runner.run("Research x")

        self.assertIn(RunTerminated, [type(event) for event in self.events])
        terminated = [e for e in self.events if type(e) is RunTerminated]
        self.assertEqual(terminated[0].reason, "max_iterations")

    async def test_stall_termination_emits_stalled(self):
        self.runner = AgentRunner(
            search_service=self.search,
            llm=self.llm,
            stall_threshold=2,
            events=self.emitter,
        )

        def _llm_response_plan(plan) -> LLMResponse:
            return _llm_response(plan)

        self.llm.generate.side_effect = [
            _llm_response_plan(PlannedQueries(queries=["q"], is_complete=False)),
            _llm_response_plan(PlannedQueries(queries=["q"], is_complete=False)),
            _llm_response("# Research Report"),
        ]
        self.search.execute_batch_search.return_value = [_search_response("q", [])]

        await self.runner.run("Research x")

        terminated = [e for e in self.events if type(e) is RunTerminated]
        self.assertEqual(terminated[0].reason, "stalled")
        self.assertEqual(sum(1 for e in self.events if type(e) is PlanCreated), 2)
        iterations = [e for e in self.events if type(e) is IterationCompleted]
        self.assertTrue(iterations)
        self.assertTrue(all(e.stalled for e in iterations))
        types = [type(e) for e in self.events]
        self.assertLess(types.index(IterationCompleted), types.index(RunTerminated))

    async def test_planning_failure_emits_termination(self):
        async def _side_effect(request):
            if "Write the final research report" in request.prompt:
                return _llm_response("# Research Report")
            raise ValueError("bad json")

        self.llm.generate.side_effect = _side_effect

        await self.runner.run("Research x")

        terminated = [e for e in self.events if type(e) is RunTerminated]
        self.assertEqual(terminated[0].reason, "planning_failed")
        iterations = [e for e in self.events if type(e) is IterationCompleted]
        self.assertEqual(len(iterations), 1)
        self.assertTrue(iterations[0].stalled)

    async def test_stalled_iteration_sets_stalled_flag(self):
        self.runner = AgentRunner(
            search_service=self.search,
            llm=self.llm,
            stall_threshold=10,
            events=self.emitter,
        )

        def _generate(request):
            prompt = request.prompt
            if "Write the final research report" in prompt:
                return _llm_response("# Research Report")
            if "URL: " in prompt:
                return _llm_response(ExtractedFindings(findings=[]))
            return _llm_response(PlannedQueries(queries=["q"], is_complete=False))

        self.llm.generate.side_effect = _generate
        self.search.execute_batch_search.return_value = [
            _search_response("q", [_result(url="https://example.com/1")])
        ]

        await self.runner.run("Research x")

        iterations = [e for e in self.events if type(e) is IterationCompleted]
        self.assertGreaterEqual(len(iterations), 1)
        self.assertTrue(iterations[0].stalled)

    async def test_concurrent_extraction_keeps_iteration_isolation(self):
        async def _generate(request):
            return _llm_response(ExtractedFindings(findings=[]))

        self.llm.generate.side_effect = _generate

        await asyncio.gather(
            self.runner._extract_findings_from_single_response(
                "q1", _search_response("q1", [_result(url="https://example.com/1")]), 1
            ),
            self.runner._extract_findings_from_single_response(
                "q2", _search_response("q2", [_result(url="https://example.com/2")]), 7
            ),
        )

        found = [e for e in self.events if type(e) is SearchResultFound]
        self.assertEqual(sorted(e.iteration for e in found), [1, 7])


if __name__ == "__main__":
    unittest.main()
