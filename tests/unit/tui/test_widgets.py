"""Unit tests for individual TUI widgets."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from textual.widgets import Collapsible, Markdown

from deep_research_agent.ai.state import Finding
from deep_research_agent.db.storage import Report
from deep_research_agent.tui.widgets import (
    CollapsibleSettings,
    FindingDetailPane,
    FindingsList,
    ReportList,
    StatusBar,
    StatusSnapshot,
)


def _dummy_finding(**overrides) -> Finding:
    defaults = {
        "title": "Test Title",
        "content": "Test content body.",
        "source_url": "https://example.com",
        "query_used": "test query",
        "extracted_at": datetime.now(UTC),
    }
    defaults.update(overrides)
    return Finding(**defaults)


def _dummy_report(**overrides) -> Report:
    defaults = {
        "id": 1,
        "topic": "Test Topic",
        "content": "# Test report",
        "created_at": datetime.now(UTC).isoformat(),
    }
    defaults.update(overrides)
    return Report(**defaults)


def _widget_text(widget) -> str:
    """Return the plain-text content of a Static/Label/StatusBar widget."""
    return str(widget.render())


# ── FindingDetailPane ─────────────────────────────────────────────────────────


class TestFindingDetailPane(unittest.IsolatedAsyncioTestCase):
    async def test_hidden_on_mount(self):
        """Widget is hidden before any finding is loaded."""
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield FindingDetailPane(id="pane")

        async with _App().run_test() as pilot:
            pane = pilot.app.query_one("#pane", FindingDetailPane)
            self.assertFalse(pane.display)

    async def test_update_shows_pane_and_sets_title(self):
        """update() makes the pane visible and renders the finding title."""
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield FindingDetailPane(id="pane")

        async with _App().run_test() as pilot:
            pane = pilot.app.query_one("#pane", FindingDetailPane)
            finding = _dummy_finding(title="My Finding", content="Important detail.")
            pane.update(finding)
            await pilot.pause()

            self.assertTrue(pane.display)
            title_text = _widget_text(pane.query_one("#detail-title"))
            self.assertIn("My Finding", title_text)

    async def test_update_sets_content_markdown(self):
        """update() populates the Markdown widget with finding content."""
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield FindingDetailPane(id="pane")

        async with _App().run_test() as pilot:
            pane = pilot.app.query_one("#pane", FindingDetailPane)
            finding = _dummy_finding(content="Very important content here.")
            pane.update(finding)
            await pilot.pause()

            md = pane.query_one("#detail-content", Markdown)
            self.assertIn("Very important content here.", md.source)

    async def test_clear_hides_pane(self):
        """clear() hides the pane after it has been shown."""
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield FindingDetailPane(id="pane")

        async with _App().run_test() as pilot:
            pane = pilot.app.query_one("#pane", FindingDetailPane)
            pane.update(_dummy_finding())
            await pilot.pause()
            self.assertTrue(pane.display)

            pane.clear()
            await pilot.pause()
            self.assertFalse(pane.display)

    async def test_clear_resets_title(self):
        """clear() wipes the title text."""
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield FindingDetailPane(id="pane")

        async with _App().run_test() as pilot:
            pane = pilot.app.query_one("#pane", FindingDetailPane)
            pane.update(_dummy_finding(title="Something"))
            await pilot.pause()
            pane.clear()
            await pilot.pause()

            title_text = _widget_text(pane.query_one("#detail-title"))
            self.assertEqual(title_text.strip(), "")


# ── FindingsList ──────────────────────────────────────────────────────────────


class TestFindingsList(unittest.IsolatedAsyncioTestCase):
    async def test_add_finding_appends_item(self):
        """add_finding() appends exactly one ListItem per call."""
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield FindingsList(id="list")

        async with _App().run_test() as pilot:
            lst = pilot.app.query_one("#list", FindingsList)
            self.assertEqual(len(lst.children), 0)

            lst.add_finding(_dummy_finding(title="Alpha"))
            await pilot.pause()
            self.assertEqual(len(lst.children), 1)

            lst.add_finding(_dummy_finding(title="Beta"))
            await pilot.pause()
            self.assertEqual(len(lst.children), 2)

    async def test_each_item_has_exactly_one_label(self):
        """Each ListItem contains a single Label (compact title-only row)."""
        from textual.app import App, ComposeResult
        from textual.widgets import Label, ListItem

        class _App(App):
            def compose(self) -> ComposeResult:
                yield FindingsList(id="list")

        async with _App().run_test() as pilot:
            lst = pilot.app.query_one("#list", FindingsList)
            lst.add_finding(_dummy_finding(title="Only Title"))
            await pilot.pause()

            item = lst.children[0]
            self.assertIsInstance(item, ListItem)
            labels = list(item.query(Label))
            self.assertEqual(len(labels), 1)
            self.assertIn("Only Title", _widget_text(labels[0]))

    async def test_findings_map_tracks_items(self):
        """findings_map maps each ListItem to its Finding."""
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield FindingsList(id="list")

        async with _App().run_test() as pilot:
            lst = pilot.app.query_one("#list", FindingsList)
            f = _dummy_finding(title="Mapped")
            lst.add_finding(f)
            await pilot.pause()

            self.assertEqual(len(lst.findings_map), 1)
            stored = next(iter(lst.findings_map.values()))
            self.assertEqual(stored.title, "Mapped")

    async def test_clear_removes_all_items_and_map(self):
        """clear() empties both the list and findings_map."""
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield FindingsList(id="list")

        async with _App().run_test() as pilot:
            lst = pilot.app.query_one("#list", FindingsList)
            lst.add_finding(_dummy_finding())
            lst.add_finding(_dummy_finding())
            await pilot.pause()
            self.assertEqual(len(lst.children), 2)

            lst.clear()
            await pilot.pause()
            self.assertEqual(len(lst.children), 0)
            self.assertEqual(len(lst.findings_map), 0)


# ── ReportList ─────────────────────────────────────────────────────────────────


class TestReportList(unittest.IsolatedAsyncioTestCase):
    def _make_app(self):
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield ReportList(id="list")

        return _App

    async def test_starts_empty(self):
        """A freshly mounted ReportList renders no rows."""
        async with self._make_app()().run_test() as pilot:
            lst = pilot.app.query_one("#list", ReportList)
            self.assertEqual(len(lst.children), 0)
            self.assertEqual(len(lst.reports_map), 0)

    async def test_set_reports_populates_list(self):
        """set_reports() renders one row per report."""
        async with self._make_app()().run_test() as pilot:
            lst = pilot.app.query_one("#list", ReportList)
            lst.set_reports([_dummy_report(topic="Alpha"), _dummy_report(id=2)])
            await pilot.pause()

            self.assertEqual(len(lst.children), 2)
            self.assertEqual(len(lst.reports_map), 2)

    async def test_set_reports_replaces_previous_items(self):
        """set_reports() replaces the list instead of appending to it."""
        async with self._make_app()().run_test() as pilot:
            lst = pilot.app.query_one("#list", ReportList)
            lst.set_reports([_dummy_report()])
            await pilot.pause()
            self.assertEqual(len(lst.children), 1)

            lst.set_reports([_dummy_report(id=2, topic="Beta")])
            await pilot.pause()

            self.assertEqual(len(lst.children), 1)
            self.assertEqual(len(lst.reports_map), 1)
            stored = next(iter(lst.reports_map.values()))
            self.assertEqual(stored.topic, "Beta")

    async def test_clear_empties_items_and_map(self):
        """clear() empties both the rendered rows and reports_map."""
        async with self._make_app()().run_test() as pilot:
            lst = pilot.app.query_one("#list", ReportList)
            lst.set_reports([_dummy_report(), _dummy_report(id=2)])
            await pilot.pause()
            self.assertEqual(len(lst.children), 2)

            lst.clear()
            await pilot.pause()
            self.assertEqual(len(lst.children), 0)
            self.assertEqual(len(lst.reports_map), 0)

    async def test_topic_is_escaped_and_truncated(self):
        """Markup in a topic is escaped and long topics are truncated to 80 chars."""
        from textual.widgets import Label

        async with self._make_app()().run_test() as pilot:
            lst = pilot.app.query_one("#list", ReportList)
            lst.set_reports([_dummy_report(topic="[bold]" + "x" * 200)])
            await pilot.pause()

            label = next(iter(lst.children[0].query(Label)))
            text = _widget_text(label)
            self.assertIn("[bold]", text)
            self.assertNotIn("**", text)
            self.assertEqual(len(text), 80)

    async def test_reports_map_tracks_items(self):
        """reports_map maps each ListItem to its Report."""
        async with self._make_app()().run_test() as pilot:
            lst = pilot.app.query_one("#list", ReportList)
            report = _dummy_report(topic="Mapped")
            lst.set_reports([report])
            await pilot.pause()

            stored = next(iter(lst.reports_map.values()))
            self.assertIs(stored, report)
            self.assertEqual(stored.content, "# Test report")


# ── CollapsibleSettings ───────────────────────────────────────────────────────


class TestCollapsibleSettings(unittest.IsolatedAsyncioTestCase):
    async def _make_app(self):
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield CollapsibleSettings(
                    initial_prompt="hello",
                    initial_max_iters=3,
                    initial_model="gpt-4o",
                    initial_provider="tavily",
                    id="settings",
                )

        return _App()

    async def test_starts_expanded(self):
        async with (await self._make_app()).run_test() as pilot:
            c = pilot.app.query_one("#settings", CollapsibleSettings).query_one(
                Collapsible
            )
            self.assertFalse(c.collapsed)

    async def test_collapse_collapses(self):
        async with (await self._make_app()).run_test() as pilot:
            s = pilot.app.query_one("#settings", CollapsibleSettings)
            s.close()
            await pilot.pause()
            c = s.query_one(Collapsible)
            self.assertTrue(c.collapsed)

    async def test_expand_expands(self):
        async with (await self._make_app()).run_test() as pilot:
            s = pilot.app.query_one("#settings", CollapsibleSettings)
            s.close()
            await pilot.pause()
            s.open()
            await pilot.pause()
            c = s.query_one(Collapsible)
            self.assertFalse(c.collapsed)

    async def test_toggle_flips_state(self):
        async with (await self._make_app()).run_test() as pilot:
            s = pilot.app.query_one("#settings", CollapsibleSettings)
            c = s.query_one(Collapsible)
            initial = c.collapsed
            s.toggle()
            await pilot.pause()
            self.assertNotEqual(c.collapsed, initial)
            s.toggle()
            await pilot.pause()
            self.assertEqual(c.collapsed, initial)


# ── StatusBar ─────────────────────────────────────────────────────────────────


class TestStatusBar(unittest.IsolatedAsyncioTestCase):
    async def _make_app(self):
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield StatusBar(id="status")

        return _App()

    async def test_idle_shows_idle_icon(self):
        async with (await self._make_app()).run_test() as pilot:
            bar = pilot.app.query_one("#status", StatusBar)
            bar.set_status(StatusSnapshot(running=False))
            await pilot.pause()
            self.assertIn("idle", _widget_text(bar))

    async def test_running_starts_timer(self):
        async with (await self._make_app()).run_test() as pilot:
            bar = pilot.app.query_one("#status", StatusBar)
            self.assertIsNone(bar._timer)
            bar.set_status(StatusSnapshot(running=True, iteration=1, max_iterations=5))
            await pilot.pause()
            self.assertIsNotNone(bar._timer)
            # Clean up timer so the test exits cleanly.
            bar.set_status(StatusSnapshot(running=False))
            await pilot.pause()

    async def test_stopping_run_stops_timer(self):
        async with (await self._make_app()).run_test() as pilot:
            bar = pilot.app.query_one("#status", StatusBar)
            bar.set_status(StatusSnapshot(running=True))
            await pilot.pause()
            self.assertIsNotNone(bar._timer)

            bar.set_status(StatusSnapshot(running=False, reason="objective complete"))
            await pilot.pause()
            self.assertIsNone(bar._timer)

    async def test_complete_reason_shows_checkmark(self):
        async with (await self._make_app()).run_test() as pilot:
            bar = pilot.app.query_one("#status", StatusBar)
            bar.set_status(StatusSnapshot(running=False, reason="objective complete"))
            await pilot.pause()
            self.assertIn("✅", _widget_text(bar))

    async def test_non_complete_reason_shows_bullet(self):
        async with (await self._make_app()).run_test() as pilot:
            bar = pilot.app.query_one("#status", StatusBar)
            bar.set_status(
                StatusSnapshot(running=False, reason="iteration budget reached")
            )
            await pilot.pause()
            self.assertIn("▪", _widget_text(bar))


if __name__ == "__main__":
    unittest.main()
