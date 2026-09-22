"""Textual-based terminal UI for the deep research agent."""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from typing import ClassVar

from rich import markup
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    ListView,
    Markdown,
    Select,
    TabbedContent,
    TabPane,
)
from textual.worker import Worker, WorkerState

from deep_research_agent.cli import DEFAULT_MODEL, build_runner
from deep_research_agent.events import (
    AgentEvent,
    EventEmitter,
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
from deep_research_agent.tui.messages import (
    RunFailedMessage,
    RunnerMessage,
    to_runner_message,
)
from deep_research_agent.tui.widgets import (
    CollapsibleSettings,
    FindingDetailPane,
    FindingsList,
    ProgressFeed,
    StatusBar,
    StatusSnapshot,
)

logger = logging.getLogger(__name__)

_DIM_OPEN = "[#808080]"
_DIM_CLOSE = "[/]"

_TERMINATION_LABELS: dict[str, str] = {
    "complete": "objective complete",
    "max_iterations": "iteration budget reached",
    "stalled": "no progress (stalled)",
    "planning_failed": "planning failed",
    "search_error": "search error",
}


def _reason_label(reason: str) -> str:
    return _TERMINATION_LABELS.get(reason, reason)


class DeepResearchApp(App):
    CSS_PATH = "app.css"
    TITLE = "Deep Research Agent"
    SUB_TITLE = "plan · search · extract · synthesize"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+s", "start_run", "Start"),
        Binding("ctrl+1", "show_tab('tab-progress')", "Progress", show=True),
        Binding("ctrl+2", "show_tab('tab-report')", "Report", show=True),
        Binding("ctrl+f", "focus_feed", "Feed", show=True),
        Binding("ctrl+x", "toggle_settings", "Settings", show=True, priority=True),
        Binding("escape", "close_detail", "Close detail", show=False),
    ]

    def __init__(
        self,
        prompt: str | None = None,
        max_iterations: int = 5,
        model: str = DEFAULT_MODEL,
        search_provider: str = "tavily",
    ) -> None:
        super().__init__()
        self._initial_prompt = prompt or ""
        self._max_iterations = max_iterations
        self._model = model
        self._search_provider = search_provider
        self._busy = False
        self._iteration = 0
        self._findings = 0
        self._reason = ""

    # ── layout ──────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="main"):
            yield CollapsibleSettings(
                initial_prompt=self._initial_prompt,
                initial_max_iters=self._max_iterations,
                initial_model=self._model,
                initial_provider=self._search_provider,
                id="settings",
            )
            with TabbedContent():
                with (
                    TabPane("📋 Progress", id="tab-progress"),
                    Vertical(id="progress-layout"),
                ):
                    yield ProgressFeed(
                        highlight=True, markup=True, wrap=True, id="feed"
                    )
                    with Horizontal(id="findings-area"):
                        yield FindingsList(id="findings")
                        yield FindingDetailPane(id="finding-detail")
                with TabPane("📄 Report", id="tab-report"):
                    yield Markdown("_No report yet._", id="report")
        yield StatusBar(id="status")
        yield Footer()

    def on_mount(self) -> None:
        if self._initial_prompt:
            self.action_start_run()
        else:
            self.query_one("#prompt", Input).focus()

    @on(ListView.Selected, "#findings")
    def _on_finding_selected(self, event: ListView.Selected) -> None:
        finding = self.query_one("#findings", FindingsList).findings_map.get(event.item)
        if finding is not None:
            self.query_one("#finding-detail", FindingDetailPane).update(finding)

    # ── actions ─────────────────────────────────────────────────────

    def action_show_tab(self, tab_id: str) -> None:
        self.query_one(TabbedContent).active = tab_id

    def action_focus_feed(self) -> None:
        self.action_show_tab("tab-progress")
        self.query_one("#feed", ProgressFeed).focus()

    def action_toggle_settings(self) -> None:
        self.query_one("#settings", CollapsibleSettings).toggle()

    def action_close_detail(self) -> None:
        detail = self.query_one("#finding-detail", FindingDetailPane)
        if detail.display:
            detail.clear()

    @on(Input.Submitted, "#prompt")
    def _on_prompt_submitted(self, _event: Input.Submitted) -> None:
        self.action_start_run()

    @on(Button.Pressed, "#start")
    def _on_start_pressed(self, _event: Button.Pressed) -> None:
        self.action_start_run()

    def _current_max_iterations(self) -> int:
        return int(self.query_one("#max-iters", Input).value.strip() or "5")

    def action_start_run(self) -> None:
        if self._busy:
            self.notify("A research run is already in progress.", severity="warning")
            return

        prompt = self.query_one("#prompt", Input).value.strip()
        if not prompt:
            self.notify("Enter a research objective first.", severity="warning")
            return

        raw_max_iters_str = self.query_one("#max-iters", Input).value.strip() or "5"
        try:
            max_iterations = int(raw_max_iters_str)
        except ValueError:
            max_iterations = -1
        if max_iterations <= 0:
            self.notify("Max iterations must be a positive integer.", severity="error")
            return

        model = self.query_one("#model", Input).value.strip() or DEFAULT_MODEL
        provider = str(self.query_one("#provider", Select).value or "tavily")

        self._busy = True
        self._iteration = 0
        self._findings = 0
        self._reason = ""
        self._set_busy(True)
        self.action_show_tab("tab-progress")

        feed = self.query_one("#feed", ProgressFeed)
        feed.clear()

        findings_list = self.query_one("#findings", FindingsList)
        findings_list.clear()

        detail = self.query_one("#finding-detail", FindingDetailPane)
        detail.clear()

        # Hide the findings area until the first finding arrives.
        self.query_one("#findings-area").display = False

        feed.add_line(f"[b cyan]▶ Running:[/] {markup.escape(prompt)}")

        self.run_worker(
            self._research(prompt, max_iterations, model, provider),
            name="research",
            group="research",
            exclusive=True,
            exit_on_error=False,
            description=prompt[:200],
        )

    async def _research(
        self,
        prompt: str,
        max_iterations: int,
        model: str,
        provider: str,
    ) -> None:
        emitter = EventEmitter()
        emitter.subscribe(self._on_agent_event)
        try:
            runner = build_runner(
                max_iterations=max_iterations,
                model=model,
                search_provider=provider,
                events=emitter,
            )
        except Exception as exc:
            logger.exception("Failed to build research runner")
            self.post_message(RunFailedMessage(f"Could not start research: {exc}"))
            return
        await runner.run(prompt)

    def _set_busy(self, busy: bool) -> None:
        for widget_id in ("prompt", "max-iters", "model", "provider"):
            self.query_one(f"#{widget_id}").disabled = busy

        start_btn = self.query_one("#start")
        start_btn.disabled = busy

        settings = self.query_one("#settings", CollapsibleSettings)
        if busy:
            settings.close()
        else:
            settings.open()

        self.query_one("#status", StatusBar).set_status(
            StatusSnapshot(
                running=busy,
                iteration=self._iteration,
                max_iterations=self._current_max_iterations(),
                findings=self._findings,
                reason=_reason_label(self._reason) if self._reason else "",
            )
        )

    def _on_agent_event(self, event: AgentEvent) -> None:
        message = to_runner_message(event)
        if message is not None:
            self.post_message(message)

    async def _set_report(self, markdown: str) -> None:
        """Render a Markdown report into the Report tab.

        A fresh widget is mounted in place of the old one rather than calling
        ``Markdown.update``: in Textual 8.2.8 the report would otherwise render
        as an empty pane (the Markdown only paints reliably when mounted into
        a visible tab, which is guaranteed by the caller).
        """
        pane = self.query_one("#tab-report", TabPane)
        old = self.query_one("#report", Markdown)
        await old.remove()
        report = Markdown(markdown, id="report")
        await pane.mount(report)
        report.scroll_home()

    # ── message handlers ────────────────────────────────────────────

    @on(RunnerMessage)
    async def _handle_runner_message(self, message: RunnerMessage) -> None:
        handler = {
            RunStarted: self._on_run_started,
            PlanCreated: self._on_plan_created,
            SearchStarted: self._on_search_started,
            SearchResultFound: self._on_search_result_found,
            SearchCompleted: self._on_search_completed,
            FindingExtracted: self._on_finding_extracted,
            IterationCompleted: self._on_iteration_completed,
            SynthesisStarted: self._on_synthesis_started,
            RunTerminated: self._on_run_terminated,
            EventError: self._on_event_error,
            ReportReady: self._on_report_ready,
        }.get(type(message.event))
        if handler is None:
            return
        result = handler(message.event)
        if isinstance(result, Awaitable):
            await result

    def _on_run_started(self, event: RunStarted) -> None:
        self.query_one("#feed", ProgressFeed).add_line(
            f"{_DIM_OPEN}🚀 research started · up to {event.max_iterations} "
            f"iteration(s){_DIM_CLOSE}"
        )

    def _on_plan_created(self, event: PlanCreated) -> None:
        feed = self.query_one("#feed", ProgressFeed)
        if event.is_complete and not event.queries:
            feed.add_line(
                f"{_DIM_OPEN}📋 \\ [iter {event.iteration}]{_DIM_CLOSE} "
                "planner: objective may be complete — checking…"
            )
        elif event.queries:
            queries = ", ".join(f"\u201c{markup.escape(q)}\u201d" for q in event.queries)
            feed.add_line(
                f"[b cyan]📋 \\ [iter {event.iteration}]{_DIM_CLOSE} "
                f"planning next queries → {queries}"
            )

    def _on_search_started(self, event: SearchStarted) -> None:
        count = len(event.queries)
        self.query_one("#feed", ProgressFeed).add_line(
            f"{_DIM_OPEN}  🔍 searching {count} quer{'y' if count == 1 else 'ies'}…{_DIM_CLOSE}"
        )

    def _on_search_result_found(self, event: SearchResultFound) -> None:
        title = markup.escape((event.result.title or event.result.url).strip())
        self.query_one("#feed", ProgressFeed).add_line(
            f"{_DIM_OPEN}    ↳ {title[:100]}{_DIM_CLOSE}"
        )

    def _on_search_completed(self, event: SearchCompleted) -> None:
        fail = f", [red]{event.failure_count} failed[/]" if event.failure_count else ""
        self.query_one("#feed", ProgressFeed).add_line(
            f"{_DIM_OPEN}  🔍 search returned {event.result_count} result(s){fail}{_DIM_CLOSE}"
        )

    def _on_finding_extracted(self, event: FindingExtracted) -> None:
        self._findings += 1
        self.query_one("#findings", FindingsList).add_finding(event.finding)
        # Reveal the findings area on the first finding.
        if self._findings == 1:
            self.query_one("#findings-area").display = True
        self.query_one("#feed", ProgressFeed).add_line(
            f"[spring_green3]    💡 finding:[/] {markup.escape(event.finding.title[:90])}"
        )
        self.query_one("#status", StatusBar).set_status(
            StatusSnapshot(
                running=True,
                iteration=self._iteration,
                max_iterations=self._current_max_iterations(),
                findings=self._findings,
            )
        )

    def _on_iteration_completed(self, event: IterationCompleted) -> None:
        self._iteration = event.iteration
        marker = "[red]stalled[/]" if event.stalled else "ok"
        self.query_one("#feed", ProgressFeed).add_line(
            f"{_DIM_OPEN}  ⏩ iteration {event.iteration} done "
            f"({marker}) · total findings: {event.total_findings}{_DIM_CLOSE}"
        )
        self.query_one("#status", StatusBar).set_status(
            StatusSnapshot(
                running=True,
                iteration=event.iteration,
                max_iterations=self._current_max_iterations(),
                findings=self._findings,
            )
        )

    def _on_synthesis_started(self, event: SynthesisStarted) -> None:
        self.query_one("#feed", ProgressFeed).add_line(
            f"{_DIM_OPEN}  ✨ synthesizing report from {event.findings_count} finding(s)…{_DIM_CLOSE}"
        )

    def _on_run_terminated(self, event: RunTerminated) -> None:
        self._reason = event.reason
        label = markup.escape(_TERMINATION_LABELS.get(event.reason, event.reason))
        icon = "✅" if event.reason == "complete" else "■"
        self.query_one("#feed", ProgressFeed).add_line(
            f"[bold orange3]{icon} ended:[/] {label}"
        )

    def _on_event_error(self, event: EventError) -> None:
        phase = markup.escape(event.phase)
        message = markup.escape(event.message)
        self.query_one("#feed", ProgressFeed).add_line(
            f"[bold red]⚠ {phase}:[/] {message}"
        )
        self.notify(f"{phase} error: {message}", severity="error")

    async def _on_report_ready(self, event: ReportReady) -> None:
        self._busy = False
        self._set_busy(False)
        # The report must be mounted while its tab is visible: content
        # mounted into a hidden TabPane is never painted once the tab is
        # shown. Switch first, then render the report into a fresh widget.
        self.action_show_tab("tab-report")
        await self._set_report(event.report)
        kind = "fallback report" if event.fallback else "report ready"
        self.query_one("#feed", ProgressFeed).add_line(
            f"[b spring_green3]✅ {kind}[/] — see the Report tab."
        )
        self.notify(
            "Research complete — review the Report tab.",
            title="Done",
            severity="information",
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "research":
            return
        if event.state == WorkerState.ERROR:
            self._busy = False
            self._set_busy(False)
            error = event.worker.error
            detail = markup.escape(str(error) if error is not None else "unknown error")
            self.query_one("#feed", ProgressFeed).add_line(
                f"[bold red]✗ run failed:[/] {detail}"
            )
            self.notify(f"Research failed: {detail}", severity="error")
        elif event.state == WorkerState.SUCCESS and self._busy:
            # Safety net: the runner finished without emitting ReportReady.
            self._busy = False
            self._set_busy(False)

    def on_run_failed_message(self, message: RunFailedMessage) -> None:
        self._busy = False
        self._set_busy(False)
        msg = markup.escape(message.message)
        self.query_one("#feed", ProgressFeed).add_line(
            f"[bold red]✗[/] {msg}"
        )
        self.notify(msg, severity="error")


def run_tui(
    prompt: str | None = None,
    max_iterations: int = 5,
    model: str = DEFAULT_MODEL,
    search_provider: str = "tavily",
) -> None:
    """Launch the Textual TUI (blocking)."""
    DeepResearchApp(
        prompt=prompt,
        max_iterations=max_iterations,
        model=model,
        search_provider=search_provider,
    ).run()
