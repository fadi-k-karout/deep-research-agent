from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """Represents an extracted piece of information with source provenance."""

    title: str
    content: str
    source_url: str
    query_used: str = Field(description="The query used to extract this finding.")
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResearchState(BaseModel):
    """Episodic state tracking the agent's progress, budget, and collected research."""

    original_prompt: str = Field(description="The user's initial research objective.")
    current_iteration: int = Field(
        default=0, description="Current execution loop count."
    )
    max_iterations: int = Field(
        default=5, description="Budget cap for loop executions."
    )

    findings: list[Finding] = Field(
        default_factory=list,
        description="Aggregated extracted findings across all searches.",
    )

    synthesized_report: str | None = Field(
        default=None, description="Final generated Markdown report."
    )
    is_complete: bool = Field(
        default=False,
        description="True only when the planner confirmed the objective is satisfied.",
    )
    termination_reason: str | None = Field(
        default=None,
        description=(
            "Why the run ended: 'complete', 'search_error', 'stalled', "
            "'planning_failed', or 'max_iterations'."
        ),
    )

    failed_searches: int = Field(
        default=0,
        description="Number of search queries that failed or returned no results.",
    )
    search_failures: list[str] = Field(
        default_factory=list,
        description="Deduplicated human-readable reasons for failed searches.",
    )
    stalled_iterations: int = Field(
        default=0,
        description="Consecutive iterations that produced no new findings.",
    )
