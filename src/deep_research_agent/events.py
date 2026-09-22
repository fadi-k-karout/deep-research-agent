"""Typed research-run events and a minimal async event emitter.

The runner emits events as a side channel so consumers (e.g. the Textual TUI,
logging, CI dashboards) can observe progress without the runner knowing
anything about them. When no subscriber is attached the emitter is a no-op and
the run behaves exactly as before.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace

from deep_research_agent.ai.state import Finding
from deep_research_agent.services.search.base import SearchResultItem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentEvent:
    """Base class for all research run events."""

    sequence: int = field(default=0, kw_only=True, compare=False)


@dataclass(frozen=True)
class RunStarted(AgentEvent):
    prompt: str
    max_iterations: int


@dataclass(frozen=True)
class PlanCreated(AgentEvent):
    iteration: int
    queries: tuple[str, ...]
    is_complete: bool


@dataclass(frozen=True)
class SearchStarted(AgentEvent):
    iteration: int
    queries: tuple[str, ...]


@dataclass(frozen=True)
class SearchCompleted(AgentEvent):
    iteration: int
    result_count: int
    failure_count: int


@dataclass(frozen=True)
class FindingExtracted(AgentEvent):
    iteration: int
    finding: Finding


@dataclass(frozen=True)
class IterationCompleted(AgentEvent):
    iteration: int
    total_findings: int
    stalled: bool


@dataclass(frozen=True)
class RunTerminated(AgentEvent):
    reason: str


@dataclass(frozen=True)
class SynthesisStarted(AgentEvent):
    findings_count: int


@dataclass(frozen=True)
class ReportReady(AgentEvent):
    report: str
    fallback: bool


@dataclass(frozen=True)
class EventError(AgentEvent):
    phase: str
    message: str


@dataclass(frozen=True)
class SearchResultFound(AgentEvent):
    """A usable search result surfaced for extraction (informational)."""

    iteration: int
    query: str
    result: SearchResultItem


Subscriber = Callable[[AgentEvent], object | Awaitable[None]]


class EventEmitter:
    """Fan-out of ``AgentEvent`` to zero or more subscribers.

    Subscribers may be sync or async callables. A failing subscriber never
    breaks the research run: errors are caught and logged.
    """

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._sequence = 0

    def subscribe(self, subscriber: Subscriber) -> None:
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)

    async def emit(self, event: AgentEvent) -> None:
        self._sequence += 1
        ev = replace(event, sequence=self._sequence)
        for subscriber in list(self._subscribers):
            try:
                result = subscriber(ev)
                if isinstance(result, Awaitable):
                    await result
            except Exception:
                logger.exception("Event subscriber failed for %s", type(ev).__name__)
