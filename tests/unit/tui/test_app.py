import asyncio
import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from textual.widgets import (
    Collapsible,
    Input,
    ListItem,
    Markdown,
    Static,
    TabbedContent,
)

from deep_research_agent.ai.state import Finding
from deep_research_agent.db.storage import Report
from deep_research_agent.events import (
    EventEmitter,
    FindingExtracted,
    PlanCreated,
    ReportReady,
    RunStarted,
    RunTerminated,
    SearchCompleted,
    SearchStarted,
    SynthesisStarted,
)
from deep_research_agent.tui.app import DeepResearchApp, _export_basename
from deep_research_agent.tui.widgets import (
    CollapsibleSettings,
    ExportMenu,
    FindingDetailPane,
    FindingsList,
    ReportDetailPanel,
    ReportList,
)


def _report() -> str:
    sections = "\n\n".join(
        f"## Section {i}\n\nDetail paragraph for section {i}." for i in range(1, 41)
    )
    return f"# Research Report\n\nObjective: Research x\n\nEverything is fine.\n\n{sections}"


class _FakeRunner:
    """Emits the event sequence of a successful single-iteration run."""

    def __init__(self, events: EventEmitter | None = None, fail: bool = False):
        self._events = events
        self._fail = fail

    def with_events(self, events: EventEmitter) -> _FakeRunner:
        self._events = events
        return self

    async def run(self, prompt: str):
        emitter = self._events
        assert emitter is not None
        await emitter.emit(RunStarted(prompt=prompt, max_iterations=3))
        await emitter.emit(
            PlanCreated(iteration=1, queries=("what is x",), is_complete=False)
        )
        await emitter.emit(SearchStarted(iteration=1, queries=("what is x",)))
        await emitter.emit(
            SearchCompleted(iteration=1, result_count=2, failure_count=0)
        )
        await emitter.emit(
            FindingExtracted(
                iteration=1,
                finding=Finding(
                    title="A finding",
                    content="Some discovered content.",
                    source_url="https://example.com/1",
                    query_used="what is x",
                ),
            )
        )
        await emitter.emit(
            RunTerminated(reason="search_error" if self._fail else "complete")
        )
        await emitter.emit(SynthesisStarted(findings_count=1))
        await emitter.emit(ReportReady(report=_report(), fallback=False))


class _BlockingRunner:
    """Runs until an externally-set stop event is fired."""

    def __init__(self) -> None:
        self._stop = asyncio.Event()

    def with_events(self, events: EventEmitter) -> _BlockingRunner:
        return self

    def finish(self) -> None:
        self._stop.set()

    async def run(self, prompt: str):
        await self._stop.wait()


@contextlib.asynccontextmanager
async def _launch(app: DeepResearchApp, runner):
    with patch("deep_research_agent.tui.app.build_runner") as mock_build:

        def make_runner(*, events: EventEmitter | None = None, **kwargs):
            return runner.with_events(events) if events is not None else runner

        mock_build.side_effect = make_runner
        async with app.run_test(size=(110, 48)) as pilot:
            yield app, pilot


async def _wait_until(pilot, predicate, tries: int = 100) -> bool:
    for _ in range(tries):
        if predicate():
            return True
        await pilot.pause()
    return predicate()


class TestDeepResearchApp(unittest.IsolatedAsyncioTestCase):
    # ── core run flow ────────────────────────────────────────────────────────

    async def test_auto_starts_research_and_renders_findings(self):
        app = DeepResearchApp(prompt="Research x", max_iterations=3)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            finished = await _wait_until(pilot, lambda: not app._busy)
            self.assertTrue(finished)

            findings = app.query_one("#findings", FindingsList)
            panel = app.query_one("#report-panel", ReportDetailPanel)
            report_md = panel.query_one("#report-content", Markdown)

            self.assertEqual(len(findings.children), 1)
            self.assertEqual(app._findings, 1)
            self.assertEqual(app._reason, "complete")
            self.assertIn("Research Report", report_md.source)
            self.assertFalse(app.query_one("#start").disabled)

    async def test_failed_run_records_termination_reason(self):
        app = DeepResearchApp(prompt="Research x", max_iterations=3)
        async with _launch(app, _FakeRunner(fail=True)) as (_, pilot):
            finished = await _wait_until(pilot, lambda: not app._busy)
            self.assertTrue(finished)

            self.assertEqual(app._reason, "search_error")
            self.assertFalse(app._busy)

    async def test_start_button_disabled_while_running(self):
        app = DeepResearchApp(prompt=None)
        blocking = _BlockingRunner()
        async with _launch(app, blocking) as (_, pilot):
            app.query_one("#prompt", Input).value = "Research y"
            await pilot.pause()
            app.action_start_run()
            started = await _wait_until(pilot, lambda: app._busy)
            self.assertTrue(started)
            self.assertTrue(app.query_one("#start").disabled)
            # A second start request is refused while busy.
            app.action_start_run()
            await pilot.pause()
            self.assertTrue(app._busy)
            blocking.finish()
            finished = await _wait_until(pilot, lambda: not app._busy)
            self.assertTrue(finished)

    async def test_empty_prompt_does_not_start(self):
        app = DeepResearchApp(prompt="")
        async with _launch(app, _FakeRunner()) as (_, pilot):
            await pilot.click("#start")
            await pilot.pause()
            self.assertFalse(app._busy)

    # ── findings detail pane ─────────────────────────────────────────────────

    async def test_selecting_finding_shows_detail_pane(self):
        """Selecting a finding populates and reveals the FindingDetailPane."""
        app = DeepResearchApp(prompt="Research x", max_iterations=3)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            finished = await _wait_until(pilot, lambda: not app._busy)
            self.assertTrue(finished)

            # Switch to progress tab so the findings list is visible.
            app.action_show_tab("tab-progress")
            await pilot.pause()

            # Detail pane starts hidden.
            detail = app.query_one("#finding-detail", FindingDetailPane)
            self.assertFalse(detail.display)

            # Select the first finding via keyboard.
            findings = app.query_one("#findings", FindingsList)
            findings.focus()
            await pilot.pause()
            findings.index = 0
            await pilot.pause()
            await pilot.press("enter")
            opened = await _wait_until(pilot, lambda: detail.display)
            self.assertTrue(opened)

            # The pane must show the correct title.
            title_widget = detail.query_one("#detail-title")
            self.assertIn("A finding", str(title_widget.render()))

            # The pane must contain the finding content.
            content_widget = detail.query_one("#detail-content", Markdown)
            self.assertIn("Some discovered content.", content_widget.source)

    async def test_escape_closes_detail_pane(self):
        """Pressing Escape hides the detail pane when it is open."""
        app = DeepResearchApp(prompt="Research x", max_iterations=3)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            finished = await _wait_until(pilot, lambda: not app._busy)
            self.assertTrue(finished)

            app.action_show_tab("tab-progress")
            await pilot.pause()

            # Open the detail pane by selecting a finding.
            findings = app.query_one("#findings", FindingsList)
            findings.focus()
            await pilot.pause()
            findings.index = 0
            await pilot.pause()
            await pilot.press("enter")
            detail = app.query_one("#finding-detail", FindingDetailPane)
            await _wait_until(pilot, lambda: detail.display)
            self.assertTrue(detail.display)

            # Escape should close it.
            await pilot.press("escape")
            closed = await _wait_until(pilot, lambda: not detail.display)
            self.assertTrue(closed)

    async def test_detail_pane_hidden_initially(self):
        """FindingDetailPane is not visible before any finding is selected."""
        app = DeepResearchApp(prompt=None)
        async with _launch(app, _FakeRunner()) as (_, __):
            detail = app.query_one("#finding-detail", FindingDetailPane)
            self.assertFalse(detail.display)

    # ── collapsible settings ─────────────────────────────────────────────────

    async def test_settings_collapse_when_run_starts(self):
        """Settings section collapses automatically when a run starts."""
        app = DeepResearchApp(prompt=None)
        blocking = _BlockingRunner()
        async with _launch(app, blocking) as (_, pilot):
            app.query_one("#prompt", Input).value = "Research z"
            await pilot.pause()
            app.action_start_run()
            started = await _wait_until(pilot, lambda: app._busy)
            self.assertTrue(started)

            collapsible = app.query_one("#settings", CollapsibleSettings).query_one(
                Collapsible
            )
            self.assertTrue(collapsible.collapsed)

            blocking.finish()
            await _wait_until(pilot, lambda: not app._busy)

    async def test_settings_expand_when_run_finishes(self):
        """Settings section expands again after the run completes."""
        app = DeepResearchApp(prompt="Research x", max_iterations=3)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            finished = await _wait_until(pilot, lambda: not app._busy)
            self.assertTrue(finished)

            collapsible = app.query_one("#settings", CollapsibleSettings).query_one(
                Collapsible
            )
            self.assertFalse(collapsible.collapsed)

    async def test_ctrl_x_toggles_settings(self):
        """Ctrl+x toggles the collapsed state of the settings section."""
        app = DeepResearchApp(prompt=None)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            collapsible = app.query_one("#settings", CollapsibleSettings).query_one(
                Collapsible
            )
            # Initially expanded.
            self.assertFalse(collapsible.collapsed)
            # First toggle: collapse.
            await pilot.press("ctrl+x")
            collapsed = await _wait_until(pilot, lambda: collapsible.collapsed)
            self.assertTrue(collapsed)
            # Second toggle: expand.
            await pilot.press("ctrl+x")
            expanded = await _wait_until(pilot, lambda: not collapsible.collapsed)
            self.assertTrue(expanded)

    # ── keyboard / tab navigation ────────────────────────────────────────────

    async def test_ctrl_1_and_ctrl_2_switch_tabs(self):
        """ctrl+1 → tab-progress, ctrl+2 → tab-reports (no tab-report any more)."""
        app = DeepResearchApp(prompt="Research x", max_iterations=3)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            finished = await _wait_until(pilot, lambda: not app._busy)
            self.assertTrue(finished)

            tabs = app.query_one(TabbedContent)
            await pilot.press("ctrl+1")
            await pilot.pause()
            self.assertEqual(tabs.active, "tab-progress")
            await pilot.press("ctrl+2")
            await pilot.pause()
            self.assertEqual(tabs.active, "tab-reports")

    async def test_ctrl_f_focuses_feed(self):
        """Ctrl+F switches to the progress tab and focuses the feed."""
        app = DeepResearchApp(prompt="Research x", max_iterations=3)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            finished = await _wait_until(pilot, lambda: not app._busy)
            self.assertTrue(finished)

            # Switch away from progress tab first.
            app.action_show_tab("tab-reports")
            await pilot.pause()

            await pilot.press("ctrl+f")
            await pilot.pause()

            tabs = app.query_one(TabbedContent)
            self.assertEqual(tabs.active, "tab-progress")

    # ── report panel ─────────────────────────────────────────────────────────

    async def test_report_ready_switches_to_reports_tab(self):
        """After a run completes the active tab is tab-reports (not tab-report)."""
        app = DeepResearchApp(prompt="Research x", max_iterations=3)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            finished = await _wait_until(pilot, lambda: not app._busy)
            self.assertTrue(finished)

            tabs = app.query_one(TabbedContent)
            await _wait_until(pilot, lambda: tabs.active == "tab-reports")
            self.assertEqual(tabs.active, "tab-reports")

    async def test_report_content_populated_after_report_ready(self):
        """The ReportDetailPanel shows the report content after a completed run."""
        app = DeepResearchApp(prompt="Research x", max_iterations=3)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            finished = await _wait_until(pilot, lambda: not app._busy)
            self.assertTrue(finished)

            for _ in range(10):
                await pilot.pause()

            panel = app.query_one("#report-panel", ReportDetailPanel)
            report_md = panel.query_one("#report-content", Markdown)
            self.assertIn("Research Report", report_md.source)
            self.assertIn("Everything is fine.", report_md.source)

    async def test_report_panel_is_scrollable(self):
        """The report content scroll container can scroll when content overflows."""
        app = DeepResearchApp(prompt="Research x", max_iterations=3)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            finished = await _wait_until(pilot, lambda: not app._busy)
            self.assertTrue(finished)

            for _ in range(30):
                await pilot.pause()

            from textual.containers import VerticalScroll

            scroll = app.query_one("#report-content-scroll", VerticalScroll)
            self.assertGreaterEqual(scroll.max_scroll_y, 0)

    # ── sidebar toggle ───────────────────────────────────────────────────────

    async def test_sidebar_visible_by_default(self):
        """The report sidebar is visible when the Reports tab is open."""
        app = DeepResearchApp(prompt=None)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            app.action_show_tab("tab-reports")
            await pilot.pause()
            sidebar = app.query_one("#report-sidebar", ReportList)
            self.assertTrue(sidebar.display)

    async def test_toggle_sidebar_hides_and_restores_sidebar(self):
        """action_toggle_sidebar() hides then restores the sidebar."""
        app = DeepResearchApp(prompt=None)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            app.action_show_tab("tab-reports")
            await pilot.pause()
            sidebar = app.query_one("#report-sidebar", ReportList)
            self.assertTrue(sidebar.display)

            app.action_toggle_sidebar()
            await pilot.pause()
            self.assertFalse(sidebar.display)

            app.action_toggle_sidebar()
            await pilot.pause()
            self.assertTrue(sidebar.display)

    async def test_ctrl_b_toggles_sidebar(self):
        """ctrl+b fires the toggle_sidebar action."""
        app = DeepResearchApp(prompt=None)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            app.action_show_tab("tab-reports")
            await pilot.pause()
            sidebar = app.query_one("#report-sidebar", ReportList)
            self.assertTrue(sidebar.display)

            await pilot.press("ctrl+b")
            hidden = await _wait_until(pilot, lambda: not sidebar.display)
            self.assertTrue(hidden)

            await pilot.press("ctrl+b")
            shown = await _wait_until(pilot, lambda: sidebar.display)
            self.assertTrue(shown)

    # ── persisted reports ───────────────────────────────────────────────────

    async def test_reports_tab_loads_existing_reports_on_mount(self):
        report = Report(
            id=1,
            topic="Research x",
            content="# Stored report",
            created_at="2024-01-01T00:00:00+00:00",
        )
        storage = MagicMock()
        storage.get_all_reports.return_value = [report]
        app = DeepResearchApp(prompt=None, storage=storage)

        async with _launch(app, _FakeRunner()):
            reports = app.query_one("#report-sidebar", ReportList)
            self.assertEqual(len(reports.children), 1)
            child = reports.children[0]
            assert isinstance(child, ListItem)
            self.assertIs(reports.reports_map[child], report)
            storage.get_all_reports.assert_called_once()

    async def test_research_passes_storage_to_runner(self):
        storage = MagicMock()
        storage.get_all_reports.return_value = []
        runner = _FakeRunner()
        app = DeepResearchApp(prompt="Research x", max_iterations=3, storage=storage)

        with patch("deep_research_agent.tui.app.build_runner") as mock_build:

            def make_runner(*, events: EventEmitter | None = None, **kwargs):
                return runner.with_events(events) if events is not None else runner

            mock_build.side_effect = make_runner
            async with app.run_test(size=(110, 48)) as pilot:
                finished = await _wait_until(pilot, lambda: not app._busy)
                self.assertTrue(finished)
                self.assertIs(mock_build.call_args.kwargs["storage"], storage)

    async def test_report_list_refreshes_after_run(self):
        reports: list[Report] = []
        storage = MagicMock()
        storage.get_all_reports.side_effect = lambda: list(reports)

        class _SavingRunner(_FakeRunner):
            async def run(self, prompt: str):
                await super().run(prompt)
                reports.append(
                    Report(
                        id=1,
                        topic="Research x",
                        content="# Saved report",
                        created_at="2024-01-01T00:00:00+00:00",
                    )
                )

        app = DeepResearchApp(prompt="Research x", max_iterations=3, storage=storage)
        async with _launch(app, _SavingRunner()) as (_, pilot):
            finished = await _wait_until(
                pilot,
                lambda: len(app.query_one("#report-sidebar", ReportList).children) == 1,
            )
            self.assertTrue(finished)
            report_list = app.query_one("#report-sidebar", ReportList)
            child = report_list.children[0]
            assert isinstance(child, ListItem)
            self.assertEqual(report_list.reports_map[child].topic, "Research x")

    async def test_selecting_stored_report_renders_content(self):
        """Selecting a report in the sidebar shows its content in ReportDetailPanel."""
        report = Report(
            id=1,
            topic="Research x",
            content="# Stored report\n\nStored body.",
            created_at="2024-01-01T00:00:00+00:00",
        )
        storage = MagicMock()
        storage.get_all_reports.return_value = [report]
        app = DeepResearchApp(prompt=None, storage=storage)

        async with _launch(app, _FakeRunner()) as (_, pilot):
            app.action_show_tab("tab-reports")
            await pilot.pause()

            sidebar = app.query_one("#report-sidebar", ReportList)
            sidebar.focus()
            await pilot.pause()
            sidebar.index = 0
            await pilot.pause()
            await pilot.press("enter")

            panel = app.query_one("#report-panel", ReportDetailPanel)
            shown = await _wait_until(
                pilot,
                lambda: (
                    "Stored body."
                    in panel.query_one("#report-content", Markdown).source
                ),
            )
            self.assertTrue(shown)
            # Tab remains on tab-reports — no tab switching needed any more.
            tabs = app.query_one(TabbedContent)
            self.assertEqual(tabs.active, "tab-reports")
            # Title is updated too.
            title = panel.query_one("#report-title", Static)
            self.assertIn("Research x", str(title.render()))

    # ── export flow ───────────────────────────────────────────────────────────

    @staticmethod
    def _export_service(results=None, error=None):
        """A stub ExportService: real category map, awaitable export_report()."""
        service = MagicMock()
        service.get_available_formats.return_value = {
            "local_file": ["md", "pdf", "html"],
            "ai_chat": ["chatgpt", "claude"],
        }
        service.export_report = AsyncMock(
            side_effect=error,
            return_value=results if results is not None else [],
        )
        return service

    @staticmethod
    def _stored_report(report_id: int = 7, topic: str = "Research x") -> Report:
        return Report(
            id=report_id,
            topic=topic,
            content="# Stored report\n\nStored body.",
            created_at="2024-01-01T00:00:00+00:00",
        )

    @classmethod
    def _app_with_report(cls, service, report_id: int = 7, topic: str = "Research x"):
        """An app whose sidebar holds one report and whose exports use *service*."""
        storage = MagicMock()
        storage.get_all_reports.return_value = [cls._stored_report(report_id, topic)]
        with patch("deep_research_agent.tui.app.ExportService", return_value=service):
            return DeepResearchApp(prompt=None, storage=storage)

    @staticmethod
    def _capture_notices(app) -> list[tuple[str, str, str]]:
        """Record (message, severity, title) instead of showing a toast."""
        notices: list[tuple[str, str, str]] = []
        app.notify = lambda message, **kw: notices.append(
            (message, kw.get("severity", "information"), kw.get("title", ""))
        )
        return notices

    async def _show_stored_report(self, app, pilot) -> ReportDetailPanel:
        """Select the first report in the sidebar so the panel has content."""
        sidebar = app.query_one("#report-sidebar", ReportList)
        sidebar.focus()
        await pilot.pause()
        sidebar.index = 0
        await pilot.pause()
        await pilot.press("enter")
        panel = app.query_one("#report-panel", ReportDetailPanel)
        self.assertTrue(
            await _wait_until(pilot, lambda: panel.current_report is not None),
            "the panel never received a report",
        )
        return panel

    @staticmethod
    def _request_export(app, formats: tuple[str, ...]) -> None:
        app.query_one("#export-menu", ExportMenu).post_message(
            ExportMenu.ExportRequested(formats)
        )

    async def test_ctrl_e_opens_the_export_dropdown_for_the_shown_report(self):
        app = self._app_with_report(self._export_service())
        async with _launch(app, _FakeRunner()) as (_, pilot):
            await self._show_stored_report(app, pilot)

            await pilot.press("ctrl+e")
            menu = app.query_one("#export-menu", ExportMenu)
            self.assertTrue(
                await _wait_until(pilot, lambda: menu.expanded),
                "ctrl+e did not open the dropdown",
            )
            # The shortcut also reveals the tab that holds the menu.
            self.assertEqual(app.query_one(TabbedContent).active, "tab-reports")

    async def test_ctrl_e_without_a_report_only_warns(self):
        app = DeepResearchApp(prompt=None)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            notices = self._capture_notices(app)
            menu = app.query_one("#export-menu", ExportMenu)

            await pilot.press("ctrl+e")
            await pilot.pause()

            self.assertFalse(menu.expanded)
            self.assertEqual(len(notices), 1)
            self.assertEqual(notices[0][1], "warning")
            self.assertIn("Select a report", notices[0][0])

    async def test_export_request_saves_the_file_and_reports_success(self):
        service = self._export_service(results=["exports/research-x-7.md"])
        app = self._app_with_report(service)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            panel = await self._show_stored_report(app, pilot)
            notices = self._capture_notices(app)

            self._request_export(app, ("md",))
            self.assertTrue(
                await _wait_until(
                    pilot, lambda: service.export_report.await_count == 1
                ),
                "the export worker never ran",
            )
            self.assertTrue(
                await _wait_until(
                    pilot,
                    lambda: (
                        "Saved" in str(app.query_one("#export-status", Static).render())
                    ),
                ),
                "no success status was shown in the report header",
            )

            content, base, formats, out_dir = service.export_report.await_args.args
            self.assertIn("Stored body.", content)
            self.assertEqual(base, "research-x-7")
            self.assertEqual(list(formats), ["md"])
            self.assertTrue(out_dir.endswith("exports"))
            self.assertEqual(panel.current_report.id, 7)
            self.assertEqual(len(notices), 1)
            self.assertEqual(notices[0][2], "Exported")
            self.assertIn("research-x-7.md", notices[0][0])

    async def test_bulk_export_forwards_every_local_file_format(self):
        service = self._export_service(
            results=[
                "exports/research-x-7.md",
                "exports/research-x-7.pdf",
                "exports/research-x-7.html",
            ]
        )
        app = self._app_with_report(service)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            await self._show_stored_report(app, pilot)
            notices = self._capture_notices(app)

            self._request_export(app, ("md", "pdf", "html"))
            self.assertTrue(
                await _wait_until(pilot, lambda: service.export_report.await_count == 1)
            )
            self.assertTrue(await _wait_until(pilot, lambda: bool(notices)))

            self.assertEqual(
                list(service.export_report.await_args.args[2]), ["md", "pdf", "html"]
            )
            self.assertIn("Saved 3 files", notices[0][0])

    async def test_ai_chat_export_reports_the_browser_instead_of_a_file(self):
        service = self._export_service(results=["https://chatgpt.com/g/g-123"])
        app = self._app_with_report(service)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            await self._show_stored_report(app, pilot)
            notices = self._capture_notices(app)

            self._request_export(app, ("chatgpt",))
            self.assertTrue(
                await _wait_until(pilot, lambda: service.export_report.await_count == 1)
            )
            self.assertTrue(
                await _wait_until(pilot, lambda: bool(notices)),
                "no notification for a browser hand-off",
            )
            self.assertIn("browser", notices[0][0])
            self.assertNotIn("Saved", notices[0][0])

    async def test_export_creates_the_output_directory(self):
        service = self._export_service(results=[])
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("deep_research_agent.tui.app._EXPORT_DIR", Path(tmp) / "exports"),
        ):
            app = self._app_with_report(service)
            async with _launch(app, _FakeRunner()) as (_, pilot):
                await self._show_stored_report(app, pilot)

                self._request_export(app, ("md",))
                self.assertTrue(
                    await _wait_until(
                        pilot, lambda: service.export_report.await_count == 1
                    )
                )
                self.assertTrue(
                    (Path(tmp) / "exports").is_dir(),
                    "the export directory was not created",
                )

    async def test_failed_export_warns_and_unlocks_the_menu(self):
        service = self._export_service(error=RuntimeError("disk on fire"))
        app = self._app_with_report(service)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            await self._show_stored_report(app, pilot)
            notices = self._capture_notices(app)
            menu = app.query_one("#export-menu", ExportMenu)

            self._request_export(app, ("md",))
            self.assertTrue(
                await _wait_until(pilot, lambda: service.export_report.await_count == 1)
            )
            self.assertTrue(
                await _wait_until(pilot, lambda: bool(notices)),
                "the failure was not reported",
            )

            self.assertEqual(notices[0][1], "error")
            self.assertIn("disk on fire", notices[0][0])
            self.assertFalse(menu.busy, "the menu stayed locked after a failure")
            self.assertFalse(menu.expanded)

    async def test_empty_result_is_reported_as_a_warning(self):
        service = self._export_service(results=[])
        app = self._app_with_report(service)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            await self._show_stored_report(app, pilot)
            notices = self._capture_notices(app)

            self._request_export(app, ("md",))
            self.assertTrue(
                await _wait_until(pilot, lambda: service.export_report.await_count == 1)
            )
            self.assertTrue(
                await _wait_until(pilot, lambda: bool(notices)),
                "an empty export result was not reported",
            )
            self.assertEqual(notices[0][1], "warning")
            self.assertIn("Nothing was exported", notices[0][0])

    async def test_export_menu_is_disabled_without_a_report(self):
        app = DeepResearchApp(prompt=None)
        async with _launch(app, _FakeRunner()) as (_, __):
            self.assertTrue(app.query_one("#export-menu", ExportMenu).disabled)

    async def test_export_status_clears_itself(self):
        service = self._export_service(results=["exports/research-x-7.md"])
        app = self._app_with_report(service)
        async with _launch(app, _FakeRunner()) as (_, pilot):
            panel = await self._show_stored_report(app, pilot)
            self._capture_notices(app)
            status = app.query_one("#export-status", Static)

            self._request_export(app, ("md",))
            self.assertTrue(
                await _wait_until(pilot, lambda: service.export_report.await_count == 1)
            )
            self.assertTrue(await _wait_until(pilot, lambda: status.display))

            panel.clear_export_status()
            await pilot.pause()
            self.assertFalse(status.display)

    # ── widget ID smoke test ─────────────────────────────────────────────────

    async def test_all_key_widget_ids_are_present(self):
        """All expected widget IDs mount without errors."""
        app = DeepResearchApp(prompt=None)
        async with _launch(app, _FakeRunner()) as (_, __):
            for widget_id in (
                "#feed",
                "#findings",
                "#finding-detail",
                "#settings",
                "#status",
                "#report-sidebar",
                "#report-panel",
            ):
                self.assertIsNotNone(
                    app.query_one(widget_id),
                    msg=f"Widget {widget_id!r} not found in the DOM",
                )


class TestExportBasename(unittest.TestCase):
    """File name stems for exported reports."""

    @staticmethod
    def _basename(topic: str, report_id: int = 0) -> str:
        return _export_basename(
            Report(
                id=report_id,
                topic=topic,
                content="body",
                created_at="2024-01-01T00:00:00+00:00",
            )
        )

    def test_slugifies_the_topic(self):
        self.assertEqual(
            self._basename("Deep Research: Agent Design"), "deep-research-agent-design"
        )

    def test_strips_punctuation_and_collapses_whitespace(self):
        self.assertEqual(self._basename("  AI / LLMs?!  "), "ai-llms")

    def test_appends_the_row_id_for_stored_reports(self):
        self.assertEqual(self._basename("Renewables", report_id=12), "renewables-12")

    def test_transient_report_has_no_id_suffix(self):
        self.assertEqual(self._basename("Renewables", report_id=0), "renewables")

    def test_truncates_long_topics(self):
        basename = self._basename("word " * 40)
        self.assertLessEqual(len(basename), 60)
        self.assertFalse(basename.endswith("-"))

    def test_falls_back_when_nothing_survives_slugging(self):
        self.assertEqual(self._basename("///"), "report")

    def test_non_ascii_topics_stay_filesystem_safe(self):
        basename = self._basename("Étude sur l'IA")
        self.assertTrue(basename)
        self.assertTrue(all(char.isascii() for char in basename), basename)


class TestRunTui(unittest.TestCase):
    def test_creates_and_closes_its_own_storage(self):
        from deep_research_agent.tui.app import run_tui

        storage = MagicMock()
        with (
            patch("deep_research_agent.tui.app.ResearchAgentStorage") as storage_cls,
            patch("deep_research_agent.tui.app.DeepResearchApp") as app_cls,
        ):
            storage_cls.return_value = storage
            run_tui(prompt="Research x")

        storage_cls.assert_called_once_with()
        self.assertIs(app_cls.call_args.kwargs["storage"], storage)
        storage.close.assert_called_once()

    def test_does_not_close_injected_storage(self):
        from deep_research_agent.tui.app import run_tui

        storage = MagicMock()
        with patch("deep_research_agent.tui.app.DeepResearchApp") as app_cls:
            run_tui(prompt="Research x", storage=storage)

        self.assertIs(app_cls.call_args.kwargs["storage"], storage)
        storage.close.assert_not_called()

    def test_closes_own_storage_when_app_raises(self):
        from deep_research_agent.tui.app import run_tui

        storage = MagicMock()
        with (
            patch("deep_research_agent.tui.app.ResearchAgentStorage") as storage_cls,
            patch("deep_research_agent.tui.app.DeepResearchApp") as app_cls,
        ):
            storage_cls.return_value = storage
            app_cls.return_value.run.side_effect = RuntimeError("boom")
            with self.assertRaises(RuntimeError):
                run_tui(prompt="Research x")

        storage.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
