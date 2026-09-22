"""Textual messages bridging research-run events to the TUI app.

The app subscribes to the runner's ``EventEmitter`` with a callback that
``post_message``-es one of these messages back to itself, keeping the runner
fully decoupled from Textual.
"""

from textual.message import Message

from deep_research_agent.events import (
    AgentEvent,
    EventError,
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


class RunnerMessage(Message):
    """Message wrapping a single agent event."""

    def __init__(self, event: AgentEvent) -> None:
        super().__init__()
        self.event = event


class RunFailedMessage(Message):
    """The research run could not start (e.g. a missing API key)."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


_KNOWN_EVENTS: tuple[type[AgentEvent], ...] = (
    RunStarted,
    PlanCreated,
    SearchStarted,
    SearchCompleted,
    SearchResultFound,
    FindingExtracted,
    IterationCompleted,
    RunTerminated,
    SynthesisStarted,
    ReportReady,
    EventError,
)


def to_runner_message(event: AgentEvent) -> RunnerMessage | None:
    """Wrap a known agent event; return None for unknown event types."""

    if not isinstance(event, _KNOWN_EVENTS):
        return None
    return RunnerMessage(event)


__all__ = [
    "RunFailedMessage",
    "RunnerMessage",
    "to_runner_message",
]
