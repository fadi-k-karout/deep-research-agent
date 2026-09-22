import asyncio
import unittest

from deep_research_agent.ai.state import Finding
from deep_research_agent.events import (
    AgentEvent,
    EventEmitter,
    FindingExtracted,
    RunStarted,
)


class TestEventEmitter(unittest.IsolatedAsyncioTestCase):
    async def test_emits_to_sync_and_async_subscribers_with_sequence(self):
        emitter = EventEmitter()
        seen: list[tuple[str, int]] = []

        def sync_subscriber(event: AgentEvent) -> None:
            seen.append(("sync", event.sequence))

        async def async_subscriber(event: AgentEvent) -> None:
            await asyncio.sleep(0)
            seen.append(("async", event.sequence))

        emitter.subscribe(sync_subscriber)
        emitter.subscribe(async_subscriber)

        await emitter.emit(RunStarted(prompt="p", max_iterations=2))
        await emitter.emit(RunStarted(prompt="q", max_iterations=1))

        self.assertEqual(len(seen), 4)
        self.assertEqual(seen[0], ("sync", 1))
        self.assertEqual(seen[1], ("async", 1))
        self.assertEqual(seen[2], ("sync", 2))
        self.assertEqual(seen[3], ("async", 2))

    async def test_failing_subscriber_does_not_break_emission(self):
        emitter = EventEmitter()
        received: list[str] = []

        def bad_subscriber(event: AgentEvent) -> None:
            raise RuntimeError("boom")

        def good_subscriber(event: AgentEvent) -> None:
            assert isinstance(event, RunStarted)
            received.append(event.prompt)

        emitter.subscribe(bad_subscriber)
        emitter.subscribe(good_subscriber)

        await emitter.emit(RunStarted(prompt="p", max_iterations=1))

        self.assertEqual(received, ["p"])

    async def test_no_subscribers_is_noop(self):
        emitter = EventEmitter()
        await emitter.emit(RunStarted(prompt="p", max_iterations=1))

    def test_events_are_frozen(self):
        event = FindingExtracted(
            iteration=1,
            finding=Finding(
                title="T", content="C", source_url="https://example.com", query_used="q"
            ),
        )
        with self.assertRaises(AttributeError):
            event.iteration = 2  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
