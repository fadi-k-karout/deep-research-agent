"""Unit tests for individual TUI widgets."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from textual.widgets import Collapsible, Markdown

from deep_research_agent.ai.state import Finding
from deep_research_agent.db.storage import Report
from deep_research_agent.tui.widgets import (
    CollapsibleSettings,
    ExportMenu,
    ExportMenuOverlay,
    FindingDetailPane,
    FindingsList,
    ReportDetailPanel,
    ReportList,
    StatusBar,
    StatusSnapshot,
)

_CATEGORIES = {
    "local_file": ["md", "pdf", "html"],
    "ai_chat": ["chatgpt", "claude"],
}


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
            await lst.set_reports([_dummy_report(topic="Alpha"), _dummy_report(id=2)])
            await pilot.pause()

            self.assertEqual(len(lst.children), 2)
            self.assertEqual(len(lst.reports_map), 2)

    async def test_set_reports_replaces_previous_items(self):
        """set_reports() replaces the list instead of appending to it."""
        async with self._make_app()().run_test() as pilot:
            lst = pilot.app.query_one("#list", ReportList)
            await lst.set_reports([_dummy_report()])
            await pilot.pause()
            self.assertEqual(len(lst.children), 1)

            await lst.set_reports([_dummy_report(id=2, topic="Beta")])
            await pilot.pause()

            self.assertEqual(len(lst.children), 1)
            self.assertEqual(len(lst.reports_map), 1)
            stored = next(iter(lst.reports_map.values()))
            self.assertEqual(stored.topic, "Beta")

    async def test_clear_empties_items_and_map(self):
        """clear() empties both the rendered rows and reports_map."""
        async with self._make_app()().run_test() as pilot:
            lst = pilot.app.query_one("#list", ReportList)
            await lst.set_reports([_dummy_report(), _dummy_report(id=2)])
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
            await lst.set_reports([_dummy_report(topic="[bold]" + "x" * 200)])
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
            await lst.set_reports([report])
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


# ── ExportMenu ─────────────────────────────────────────────────────────────────


class TestExportMenu(unittest.IsolatedAsyncioTestCase):
    def _make_app(self, categories=None):
        from textual import on
        from textual.app import App, ComposeResult
        from textual.containers import Vertical

        chosen = _CATEGORIES if categories is None else categories

        class _App(App):
            DEFAULT_CSS = "#right { align: right middle; }"

            def __init__(self) -> None:
                super().__init__()
                self.requests: list[tuple[str, ...]] = []

            def compose(self) -> ComposeResult:
                # The menu lives in the right-hand edge of the report header.
                with Vertical(id="right"):
                    yield ExportMenu(chosen, id="menu")

            @on(ExportMenu.ExportRequested)
            def _record(self, event: ExportMenu.ExportRequested) -> None:
                self.requests.append(event.formats)

        return _App()

    @staticmethod
    def _parts(app) -> ExportMenu:
        return app.query_one("#menu", ExportMenu)

    @staticmethod
    def _overlay(app) -> ExportMenuOverlay:
        return app.query_one(ExportMenuOverlay)

    @staticmethod
    def _trigger(app):
        from textual.widgets import Static

        return app.query_one("#menu .export-trigger", Static)

    def _options(self, app) -> list[tuple[str | None, bool]]:
        overlay = self._overlay(app)
        return [
            (overlay.get_option_at_index(i).id, overlay.get_option_at_index(i).disabled)
            for i in range(overlay.option_count)
        ]

    async def test_starts_collapsed_with_a_hidden_overlay(self):
        async with self._make_app().run_test() as pilot:
            app = pilot.app
            self.assertFalse(self._parts(app).expanded)
            self.assertFalse(self._overlay(app).display)

    async def test_disabled_until_a_report_is_on_screen(self):
        async with self._make_app().run_test() as pilot:
            app = pilot.app
            menu = self._parts(app)
            self.assertTrue(menu.disabled, "no formats loaded yet")
            menu.set_formats(_CATEGORIES)
            self.assertTrue(menu.disabled, "no report on screen yet")

            menu.set_report_available(True)
            self.assertFalse(menu.disabled)

    async def test_stays_disabled_when_there_are_no_formats(self):
        async with self._make_app({}).run_test() as pilot:
            app = pilot.app
            menu = self._parts(app)
            menu.set_report_available(True)
            self.assertTrue(menu.disabled)

    async def test_options_are_grouped_by_category_with_disabled_headers(self):
        async with self._make_app().run_test() as pilot:
            app = pilot.app
            options = self._options(app)

            # Headers are the disabled (None) entries; the blank lines between
            # them are OptionList separators, which are not addressable by index.
            self.assertEqual(
                options,
                [
                    (None, True),  # "Save to file" header
                    ("md", False),
                    ("pdf", False),
                    ("html", False),
                    (None, True),  # "Open in AI chat" header
                    ("chatgpt", False),
                    ("claude", False),
                    ("all", False),
                ],
            )

    async def test_bulk_entry_is_absent_for_a_single_local_file_format(self):
        async with self._make_app({"local_file": ["md"]}).run_test() as pilot:
            app = pilot.app
            self.assertEqual(
                [option_id for option_id, _ in self._options(app)],
                [None, "md"],
            )

    async def test_enter_expands_and_moves_focus_to_the_overlay(self):
        async with self._make_app().run_test() as pilot:
            app = pilot.app
            menu = self._parts(app)
            menu.set_report_available(True)
            menu.focus()
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()

            self.assertTrue(menu.expanded)
            self.assertTrue(self._overlay(app).display)
            self.assertTrue(self._overlay(app).has_focus)

    async def test_highlight_starts_on_the_first_selectable_option(self):
        """Category headers are disabled, so the first option is selected."""
        async with self._make_app().run_test() as pilot:
            app = pilot.app
            menu = self._parts(app)
            menu.set_report_available(True)
            menu.expanded = True
            await pilot.pause()

            self.assertEqual(self._overlay(app).highlighted, 1)

    async def test_arrow_keys_skip_disabled_headers(self):
        async with self._make_app().run_test() as pilot:
            app = pilot.app
            menu = self._parts(app)
            menu.set_report_available(True)
            menu.expanded = True
            await pilot.pause()

            overlay = self._overlay(app)
            await pilot.press("down")
            await pilot.pause()

            self.assertEqual(overlay.highlighted, 2)

    async def test_escape_closes_the_menu_and_returns_focus(self):
        async with self._make_app().run_test() as pilot:
            app = pilot.app
            menu = self._parts(app)
            menu.set_report_available(True)
            menu.expanded = True
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            self.assertFalse(menu.expanded)
            self.assertFalse(self._overlay(app).display)
            self.assertTrue(menu.has_focus)

    async def test_clicking_the_trigger_toggles_the_menu(self):
        async with self._make_app().run_test() as pilot:
            app = pilot.app
            menu = self._parts(app)
            menu.set_report_available(True)
            await pilot.pause()

            await pilot.click("#menu .export-trigger")
            await pilot.pause()
            self.assertTrue(menu.expanded)

            await pilot.click("#menu .export-trigger")
            await pilot.pause()
            self.assertFalse(menu.expanded)

    async def test_trigger_label_reflects_the_open_state(self):
        async with self._make_app().run_test() as pilot:
            app = pilot.app
            menu = self._parts(app)
            menu.set_report_available(True)
            await pilot.pause()

            self.assertEqual(str(self._trigger(app).render()), "⇩ Export ▾")

            menu.expanded = True
            await pilot.pause()
            self.assertEqual(str(self._trigger(app).render()), "⇩ Export ▴")

    async def test_choosing_a_format_posts_an_export_request(self):
        async with self._make_app().run_test() as pilot:
            app = pilot.app
            menu = self._parts(app)
            menu.set_report_available(True)
            menu.expanded = True
            await pilot.pause()

            overlay = self._overlay(app)
            overlay.highlighted = 2  # PDF
            await pilot.press("enter")
            await pilot.pause()

            self.assertEqual(app.requests, [("pdf",)])
            self.assertFalse(menu.expanded, "the menu closes after a choice")

    async def test_choosing_save_all_requests_every_local_file_format(self):
        async with self._make_app().run_test() as pilot:
            app = pilot.app
            menu = self._parts(app)
            menu.set_report_available(True)
            menu.expanded = True
            await pilot.pause()

            overlay = self._overlay(app)
            overlay.highlighted = overlay.option_count - 1
            await pilot.press("enter")
            await pilot.pause()

            self.assertEqual(app.requests, [("md", "pdf", "html")])

    async def test_clicking_an_option_exports_it(self):
        async with self._make_app().run_test(size=(120, 30)) as pilot:
            app = pilot.app
            menu = self._parts(app)
            menu.set_report_available(True)
            menu.expanded = True
            for _ in range(4):
                await pilot.pause()

            # Row 0 is the "Save to file" header, so "HTML" sits on content row 3.
            await pilot.click(ExportMenuOverlay, offset=(4, 1 + 3))
            await pilot.pause()

            self.assertEqual(app.requests, [("html",)])

    async def test_menu_cannot_expand_while_disabled(self):
        async with self._make_app().run_test() as pilot:
            app = pilot.app
            menu = self._parts(app)
            self.assertTrue(menu.disabled)

            menu.expanded = True
            await pilot.pause()

            self.assertFalse(menu.expanded)

    async def test_busy_label_and_input_lock(self):
        async with self._make_app().run_test() as pilot:
            app = pilot.app
            menu = self._parts(app)
            menu.set_report_available(True)
            await pilot.pause()

            menu.busy = True
            await pilot.pause()
            self.assertEqual(str(self._trigger(app).render()), "⏳ Exporting…")

            menu.action_toggle_menu()
            await pilot.pause()
            self.assertFalse(menu.expanded, "no second export mid-flight")

            menu.busy = False
            await pilot.pause()
            self.assertEqual(str(self._trigger(app).render()), "⇩ Export ▾")

    async def test_overlay_is_right_aligned_with_the_trigger(self):
        """The panel hangs off the button instead of running off the screen."""
        async with self._make_app().run_test(size=(100, 24)) as pilot:
            app = pilot.app
            menu = self._parts(app)
            menu.set_report_available(True)
            menu.expanded = True
            for _ in range(4):
                await pilot.pause()

            overlay = self._overlay(app)
            self.assertEqual(
                overlay.region.x + overlay.region.width,
                menu.region.x + menu.region.width,
            )


# ── ReportDetailPanel ──────────────────────────────────────────────────────────


class TestReportDetailPanel(unittest.IsolatedAsyncioTestCase):
    def _make_app(self):
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield ReportDetailPanel(id="panel")

        return _App

    async def test_starts_in_placeholder_state(self):
        """Panel shows placeholder markdown and empty title on mount."""
        async with self._make_app()().run_test() as pilot:
            panel = pilot.app.query_one("#panel", ReportDetailPanel)
            from textual.widgets import Markdown, Static

            md = panel.query_one("#report-content", Markdown)
            title = panel.query_one("#report-title", Static)
            self.assertIn("Select a report", md.source)
            self.assertEqual(str(title.render()).strip(), "")

    async def test_show_sets_title_and_content(self):
        """show() updates both the title Static and the Markdown content."""
        async with self._make_app()().run_test() as pilot:
            panel = pilot.app.query_one("#panel", ReportDetailPanel)
            from textual.widgets import Markdown, Static

            report = _dummy_report(topic="My Topic", content="# Hello\n\nBody text.")
            panel.show(report)
            await pilot.pause()

            title = panel.query_one("#report-title", Static)
            md = panel.query_one("#report-content", Markdown)
            self.assertIn("My Topic", str(title.render()))
            self.assertIn("# Hello", md.source)
            self.assertIn("Body text.", md.source)

    async def test_clear_resets_title_and_content(self):
        """clear() empties the title and restores the placeholder."""
        async with self._make_app()().run_test() as pilot:
            panel = pilot.app.query_one("#panel", ReportDetailPanel)
            from textual.widgets import Markdown, Static

            panel.show(_dummy_report(topic="Something", content="# Content"))
            await pilot.pause()

            panel.clear()
            await pilot.pause()

            title = panel.query_one("#report-title", Static)
            md = panel.query_one("#report-content", Markdown)
            self.assertEqual(str(title.render()).strip(), "")
            self.assertIn("Select a report", md.source)

    async def test_toggle_button_is_present(self):
        """The ≡ toggle button is composed inside the panel."""
        from textual.widgets import Button

        async with self._make_app()().run_test() as pilot:
            panel = pilot.app.query_one("#panel", ReportDetailPanel)
            btn = panel.query_one("#toggle-sidebar", Button)
            self.assertIsNotNone(btn)

    # ── export dropdown wiring ────────────────────────────────────────────

    async def test_current_report_tracks_show_and_clear(self):
        """current_report is the report on screen, or None when empty."""
        async with self._make_app()().run_test() as pilot:
            panel = pilot.app.query_one("#panel", ReportDetailPanel)
            self.assertIsNone(panel.current_report)

            report = _dummy_report()
            panel.show(report)
            self.assertIs(panel.current_report, report)

            panel.clear()
            self.assertIsNone(panel.current_report)

    async def test_export_menu_follows_the_loaded_report(self):
        """The dropdown is enabled only while a report is displayed."""
        async with self._make_app()().run_test() as pilot:
            panel = pilot.app.query_one("#panel", ReportDetailPanel)
            panel.set_export_formats(_CATEGORIES)
            menu = panel.query_one("#export-menu", ExportMenu)
            await pilot.pause()

            self.assertTrue(menu.disabled)

            panel.show(_dummy_report())
            await pilot.pause()
            self.assertFalse(menu.disabled)

            panel.clear()
            await pilot.pause()
            self.assertTrue(menu.disabled)

    async def test_export_menu_offers_the_service_formats(self):
        async with self._make_app()().run_test() as pilot:
            panel = pilot.app.query_one("#panel", ReportDetailPanel)
            panel.set_export_formats(_CATEGORIES)
            panel.show(_dummy_report())
            await pilot.pause()

            overlay = panel.query_one(ExportMenuOverlay)
            ids = [
                overlay.get_option_at_index(i).id for i in range(overlay.option_count)
            ]
            self.assertEqual(
                ids, [None, "md", "pdf", "html", None, "chatgpt", "claude", "all"]
            )

    async def test_export_status_is_hidden_until_shown(self):
        """The inline status takes no room in the top bar while empty."""
        from textual.widgets import Static

        async with self._make_app()().run_test() as pilot:
            panel = pilot.app.query_one("#panel", ReportDetailPanel)
            status = panel.query_one("#export-status", Static)
            await pilot.pause()
            self.assertFalse(status.display)

            panel.show_export_status("✓ Saved exports/report-1.md")
            await pilot.pause()
            self.assertTrue(status.display)
            self.assertIn("exports/report-1.md", str(status.render()))

            panel.clear_export_status()
            await pilot.pause()
            self.assertFalse(status.display)
            self.assertEqual(str(status.render()).strip(), "")

    async def test_clear_also_hides_the_export_status(self):
        from textual.widgets import Static

        async with self._make_app()().run_test() as pilot:
            panel = pilot.app.query_one("#panel", ReportDetailPanel)
            panel.show(_dummy_report())
            panel.show_export_status("✓ Saved exports/report-1.md")
            await pilot.pause()

            panel.clear()
            await pilot.pause()

            self.assertFalse(panel.query_one("#export-status", Static).display)


if __name__ == "__main__":
    unittest.main()
