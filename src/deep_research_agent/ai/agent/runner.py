import asyncio
import logging

from pydantic import BaseModel, Field

from deep_research_agent.ai.llm.base import BaseLLMProvider, LLMRequest
from deep_research_agent.ai.state import Finding, ResearchState
from deep_research_agent.services.search.base import (
    SearchResponse,
    SearchResultItem,
)
from deep_research_agent.services.search.service import SearchService

logger = logging.getLogger(__name__)

_MAX_RESULT_CHARS = 8000
_MAX_FINDING_CHARS = 2000

_PLANNING_SYSTEM_PROMPT = (
    "You are a research planner. Given a research objective and the findings "
    "gathered so far, decide whether more web searches are needed to answer it. "
    "If more research is required, propose focused search queries that fill the "
    "gaps in the current findings. Propose at most 3 queries at a time. Return "
    "your decision as structured JSON."
)

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a research analyst. Extract concise, factual findings from a web "
    "search result. Each finding must be a self-contained statement useful for "
    "a research report. The 'findings' field must be a JSON array of objects, "
    "where every object has a string 'title' and a string 'content'. Do not "
    "include empty strings, nulls, or entries of any other type. Return your "
    "findings as structured JSON."
)

_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a research writer. Synthesize the provided findings into a clear, "
    "well-structured Markdown report that directly answers the research "
    "objective. Cite every factual claim with its source URL."
)


class PlannedQueries(BaseModel):
    """Structured planning output: the next search queries and completion signal."""

    queries: list[str] = Field(
        default_factory=list,
        description="Next search queries to run. Empty when no more research is needed.",
    )
    is_complete: bool = Field(
        default=False,
        description="True when the accumulated findings satisfy the objective.",
    )


class FindingDraft(BaseModel):
    """A raw extracted finding before provenance is attached."""

    title: str = Field(description="Short title of the finding.")
    content: str = Field(description="The extracted information.")


class ExtractedFindings(BaseModel):
    """Structured extraction output for a single search result."""

    findings: list[FindingDraft] = Field(
        default_factory=list, description="Findings extracted from one result."
    )


class AgentRunner:
    def __init__(
        self,
        search_service: SearchService,
        llm: BaseLLMProvider,
        max_iterations: int = 5,
        stall_threshold: int = 2,
    ):
        self.search_service = search_service
        self.llm = llm
        self.max_iterations = max_iterations
        self.stall_threshold = stall_threshold

    async def run(self, prompt: str) -> ResearchState:
        """Main loop of the agent runner."""
        state = ResearchState(
            original_prompt=prompt,
            max_iterations=self.max_iterations,
        )

        logger.info(f"Running agent with prompt: {prompt}")

        while not state.is_complete and state.current_iteration < state.max_iterations:
            state.current_iteration += 1
            logger.info(f"Current iteration: {state.current_iteration}")

            queries = await self._plan_next_actions(state)

            if not queries:
                logger.info("No actions planned, stopping.")
                state.is_complete = True
                break

            try:
                search_response = await self.search_service.execute_batch_search(
                    queries
                )
            except Exception:
                logger.exception("Error during search; stopping.")
                state.is_complete = True
                break

            self._record_search_outcomes(state, queries, search_response)

            findings = await self._extract_findings(queries, search_response)
            if findings:
                state.findings.extend(findings)
                state.stalled_iterations = 0
                logger.info(
                    f"Collected {len(findings)} new findings "
                    f"(total {len(state.findings)})."
                )
            else:
                state.stalled_iterations += 1
                logger.info(
                    f"Iteration added no findings "
                    f"({state.stalled_iterations} consecutive stalled)."
                )
                if state.stalled_iterations >= self.stall_threshold:
                    logger.warning("No research progress; stopping early.")
                    state.is_complete = True
                    break

        state.synthesized_report = await self._synthesize_report(state)
        state.is_complete = True
        logger.info("Research complete.")
        return state

    def _record_search_outcomes(
        self,
        state: ResearchState,
        queries: list[str],
        responses: list[SearchResponse],
    ) -> None:
        """Record failed or empty searches so the report can explain gaps."""
        for query, response in zip(queries, responses):
            if response.results:
                continue
            state.failed_searches += 1
            if not response.success:
                message = f"{query}: {response.error or 'search failed'}"
            else:
                message = f"{query}: no results returned"
            if message not in state.search_failures:
                state.search_failures.append(message)

    async def _plan_next_actions(self, state: ResearchState) -> list[str]:
        """Decide the next queries based on the current findings.

        Marks the state complete when the planner determines the objective has
        been satisfied. Retries once on LLM failure before stopping defensively.
        """
        for attempt in range(2):
            try:
                plan = await self._generate_plan(state)
                state.is_complete = plan.is_complete
                return plan.queries
            except Exception:
                if attempt == 0:
                    logger.warning("Planning failed; retrying once.")
                    continue
                logger.exception("Planning failed after retry; stopping research.")

        state.is_complete = True
        return []

    async def _generate_plan(self, state: ResearchState) -> PlannedQueries:
        response = await self.llm.generate(
            LLMRequest(
                prompt=self._build_plan_prompt(state),
                system_prompt=_PLANNING_SYSTEM_PROMPT,
                response_model=PlannedQueries,
                temperature=0.7,
            )
        )
        return response.content

    async def _extract_findings_from_single_response(
        self, query: str, search_response: SearchResponse
    ) -> list[Finding]:
        """Extract findings from the current query's results, in parallel."""
        findings: list[Finding] = []

        if not search_response.success or not search_response.results:
            logger.info(f"Skipping failed or empty response for query: {query}")
            return findings

        batches = await asyncio.gather(
            *(
                self._extract_from_result(query, result)
                for result in search_response.results
            )
        )
        for batch in batches:
            findings.extend(batch)
        return findings

    async def _extract_from_result(
        self, query: str, result: SearchResultItem
    ) -> list[Finding]:
        """Extract findings from a single search result, retrying once on failure."""
        for attempt in range(2):
            try:
                response = await self.llm.generate(
                    LLMRequest(
                        prompt=self._build_extraction_prompt(result),
                        system_prompt=_EXTRACTION_SYSTEM_PROMPT,
                        response_model=ExtractedFindings,
                    )
                )
                return [
                    Finding(
                        title=draft.title,
                        content=draft.content,
                        source_url=result.url,
                        query_used=query,
                    )
                    for draft in response.content.findings
                ]
            except Exception:
                if attempt == 0:
                    logger.warning(
                        "Extraction failed for result: %s; retrying once.",
                        result.url,
                    )
                    continue
                logger.exception(
                    "Extraction failed for result: %s; skipping.", result.url
                )

        return []

    async def _extract_findings(
        self, queries: list[str], search_response: list[SearchResponse]
    ) -> list[Finding]:
        """Extract findings from the current query/queries results."""
        findings: list[Finding] = []
        for query, response in zip(queries, search_response):
            findings.extend(
                await self._extract_findings_from_single_response(query, response)
            )
        return findings

    async def _synthesize_report(self, state: ResearchState) -> str:
        """Summarize all accumulated findings into a Markdown formatted report.

        When no findings were gathered, synthesize nothing: a fallback report
        explains why instead of fabricating an empty narrative.
        """
        if not state.findings:
            report = self._build_fallback_report(state)
            state.synthesized_report = report
            return report

        try:
            response = await self.llm.generate(
                LLMRequest(
                    prompt=self._build_synthesis_prompt(state),
                    system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
                    temperature=0.4,
                )
            )
        except Exception:
            logger.exception("Report synthesis failed; using fallback report.")
            report = self._build_fallback_report(state)
            state.synthesized_report = report
            return report

        report = response.content
        if state.failed_searches:
            report = f"{report}\n\n{self._build_research_notes(state)}"
        state.synthesized_report = report
        return report

    def _build_plan_prompt(self, state: ResearchState) -> str:
        findings = self._format_findings(state)
        return (
            f"Research objective:\n{state.original_prompt}\n\n"
            f"Findings gathered so far:\n{findings or 'None yet.'}\n\n"
            "Are more searches needed to answer the objective? If so, propose "
            "the next search queries."
        )

    def _build_extraction_prompt(self, result: SearchResultItem) -> str:
        content = result.content
        if len(content) > _MAX_RESULT_CHARS:
            content = content[:_MAX_RESULT_CHARS] + "\u2026"
        return (
            f"Title: {result.title}\n"
            f"URL: {result.url}\n"
            f"Content:\n{content}\n\n"
            "Extract the key findings from this search result."
        )

    def _build_synthesis_prompt(self, state: ResearchState) -> str:
        findings = self._format_findings(state)
        prompt = (
            f"Research objective:\n{state.original_prompt}\n\n"
            f"Findings:\n{findings}\n\n"
            "Write the final research report in Markdown."
        )
        if state.search_failures:
            failures = "\n".join(f"- {message}" for message in state.search_failures)
            prompt += (
                "\n\nSome searches failed or returned no results. Acknowledge "
                f"these limitations where relevant:\n{failures}"
            )
        return prompt

    def _format_findings(self, state: ResearchState) -> str:
        lines = []
        for index, finding in enumerate(state.findings, start=1):
            content = finding.content
            if len(content) > _MAX_FINDING_CHARS:
                content = content[:_MAX_FINDING_CHARS] + "\u2026"
            lines.append(
                f"{index}. {finding.title}\n"
                f"   {content}\n"
                f"   Source: {finding.source_url}"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_fallback_report(state: ResearchState) -> str:
        lines = [
            f"# Research Report\n\nObjective: {state.original_prompt}",
        ]
        if not state.findings:
            lines.append("\nNo findings were gathered during this research session.")
            if state.search_failures:
                lines.append("\nThe following searches failed or returned no results:")
                for message in state.search_failures:
                    lines.append(f"- {message}")
            elif state.stalled_iterations:
                lines.append(
                    "\nNo usable findings were gathered across the research iterations."
                )
            else:
                lines.append(
                    "\nThe planner ended the session before any findings were gathered."
                )
        else:
            for index, finding in enumerate(state.findings, start=1):
                lines.append(
                    f"\n## {index}. {finding.title}\n\n"
                    f"{finding.content}\n\nSource: {finding.source_url}"
                )
            if state.search_failures:
                lines.append(f"\n\n{AgentRunner._build_research_notes(state)}")
        return "".join(lines)

    @staticmethod
    def _build_research_notes(state: ResearchState) -> str:
        lines = [
            "## Research Notes",
            "",
            f"{state.failed_searches} search(es) failed or returned no results ",
            "during this session:",
        ]
        for message in state.search_failures:
            lines.append(f"- {message}")
        return "\n".join(lines)
