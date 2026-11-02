"""Custom widgets for the deep research TUI."""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from rich.markup import escape
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.reactive import var
from textual.timer import Timer
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    OptionList,
    RichLog,
    Select,
    Static,
)
from textual.widgets.option_list import Option

from deep_research_agent.ai.state import Finding
from deep_research_agent.db.storage import Report

# ── Spinner frames ────────────────────────────────────────────────────────────

_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


# ── FindingDetailPane ─────────────────────────────────────────────────────────


class FindingDetailPane(VerticalScroll):
    """Right-side pane that renders the full content of a selected Finding.

    Hidden by default; call ``update(finding)`` to populate and show it,
    ``clear()`` to hide it again.
    """

    DEFAULT_CSS = """
    FindingDetailPane {
        display: none;
        border-left: solid $border;
        padding: 1 2;
        width: 2fr;
    }
    FindingDetailPane #detail-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    FindingDetailPane #detail-meta {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="detail-title")
        yield Markdown("", id="detail-content")
        yield Static("", id="detail-meta")

    def on_mount(self) -> None:
        self.display = False

    def update(self, finding: Finding) -> None:  # type: ignore[override]
        """Populate the pane with *finding* and make it visible."""
        self.query_one("#detail-title", Static).update(f"💡 {finding.title}")
        self.query_one("#detail-content", Markdown).update(finding.content)
        meta_lines = "\n".join(
            [
                f"🔗 {finding.source_url}",
                f"🔍 {finding.query_used}",
                f"🕒 {finding.extracted_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            ]
        )
        self.query_one("#detail-meta", Static).update(meta_lines)
        self.scroll_home(animate=False)
        self.display = True

    def clear(self) -> None:
        """Hide the pane and reset its contents."""
        self.display = False
        self.query_one("#detail-title", Static).update("")
        self.query_one("#detail-content", Markdown).update("")
        self.query_one("#detail-meta", Static).update("")


# ── FindingsList ──────────────────────────────────────────────────────────────


class FindingsList(ListView):
    """Compact live-updating list of extracted findings — one title per row."""

    DEFAULT_CSS = """
    FindingsList {
        width: 1fr;
        border: none;
    }
    FindingsList > ListItem {
        padding: 0 1;
        height: 1;
    }
    FindingsList > ListItem.--highlight {
        border-left: thick $accent;
        background: $boost;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.findings_map: dict[ListItem, Finding] = {}

    def add_finding(self, finding: Finding) -> None:
        item = ListItem(
            Label(f"💡 {finding.title}", classes="finding-title"),
            classes="finding-item",
        )
        self.findings_map[item] = finding
        self.append(item)
        self.scroll_end(animate=False)

    def clear(self) -> None:  # type: ignore[override]
        self.findings_map.clear()
        super().clear()


# ── ProgressFeed ──────────────────────────────────────────────────────────────


class ProgressFeed(RichLog):
    """Scrollable chronological log of the research run."""

    DEFAULT_CSS = """
    ProgressFeed {
        border: round $primary;
        padding: 0 2;
        height: 1fr;
    }
    """

    def add_line(self, text: str) -> None:
        self.write(text)


# ── CollapsibleSettings ───────────────────────────────────────────────────────


class CollapsibleSettings(Vertical):
    """Settings section that can be collapsed/expanded.

    Contains two rows:
      - Row 1: prompt input + Start button
      - Row 2: max-iters + model + provider
    """

    DEFAULT_CSS = """
    CollapsibleSettings {
        height: auto;
        background: $panel;
        border-bottom: solid $border;
    }
    CollapsibleSettings Collapsible {
        border: none;
        padding: 0;
        background: transparent;
    }
    CollapsibleSettings CollapsibleTitle {
        padding: 0 1;
        color: $text-muted;
        background: $panel;
    }
    CollapsibleSettings CollapsibleTitle:focus {
        background: $panel;
        border: none;
        color: $text-muted;
    }
    CollapsibleSettings CollapsibleTitle:hover {
        background: $boost;
        color: $text;
    }
    CollapsibleSettings #settings-row-main {
        height: 3;
        padding: 0 1;
        align-horizontal: left;
    }
    CollapsibleSettings #settings-row-options {
        height: 3;
        padding: 0 1;
        align-horizontal: left;
    }
    CollapsibleSettings .field-group {
        height: 3;
        align-vertical: middle;
        width: auto;
        margin-right: 2;
    }
    CollapsibleSettings .field-label {
        color: $text-muted;
        width: auto;
        margin-right: 1;
        content-align: left middle;
        height: 3;
    }
    CollapsibleSettings #prompt {
        width: 1fr;
        margin-right: 1;
    }
    CollapsibleSettings #max-iters {
        width: 8;
    }
    CollapsibleSettings #model {
        width: 40;
    }
    CollapsibleSettings #provider {
        width: 24;
    }
    """

    def __init__(
        self,
        initial_prompt: str = "",
        initial_max_iters: int = 5,
        initial_model: str = "",
        initial_provider: str = "tavily",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._initial_prompt = initial_prompt
        self._initial_max_iters = initial_max_iters
        self._initial_model = initial_model
        self._initial_provider = initial_provider

    def compose(self) -> ComposeResult:
        with Collapsible(title="⚙ Settings", collapsed=False):
            with Horizontal(id="settings-row-main"):
                yield Input(
                    placeholder="Research objective…",
                    value=self._initial_prompt,
                    id="prompt",
                )
                yield Button("Start", variant="primary", id="start")
            with Horizontal(id="settings-row-options"):
                with Horizontal(classes="field-group"):
                    yield Label("Max Iterations:", classes="field-label")
                    yield Input(
                        value=str(self._initial_max_iters),
                        type="integer",
                        id="max-iters",
                        placeholder="5",
                    )
                with Horizontal(classes="field-group"):
                    yield Label("Model:", classes="field-label")
                    yield Input(
                        value=self._initial_model, id="model", placeholder="Model"
                    )
                with Horizontal(classes="field-group"):
                    yield Label("Search Provider:", classes="field-label")
                    yield Select(
                        [("Tavily", "tavily"), ("Exa", "exa")],
                        allow_blank=False,
                        value=self._initial_provider,
                        id="provider",
                    )

    def close(self) -> None:
        """Collapse the settings section."""
        self.query_one(Collapsible).collapsed = True

    def open(self) -> None:
        """Expand the settings section."""
        self.query_one(Collapsible).collapsed = False

    def toggle(self) -> None:
        """Toggle the collapsed state."""
        c = self.query_one(Collapsible)
        c.collapsed = not c.collapsed


# ── StatusBar ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StatusSnapshot:
    """Value object for StatusBar state; travels as one unit."""

    running: bool = False
    iteration: int = 0
    max_iterations: int = 0
    findings: int = 0
    reason: str = ""


class StatusBar(Static):
    """Single-line status with an animated spinner during a run."""

    DEFAULT_CSS = """
    StatusBar {
        height: 3;
        padding: 0 2;
        background: $panel;
        border-top: solid $border;
        content-align: left middle;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._frame: int = 0
        self._timer = None
        self._snapshot: StatusSnapshot = StatusSnapshot()

    def set_status(self, snapshot: StatusSnapshot | None = None) -> None:
        current = snapshot if snapshot is not None else StatusSnapshot()
        self._snapshot = current

        was_running = self._timer is not None

        if current.running and not was_running:
            self._timer = self.set_interval(0.1, self._tick)
        elif not current.running and was_running and self._timer is not None:
            self._timer.stop()
            self._timer = None
            self._frame = 0

        if not current.running:
            self._render_status(current)

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_SPINNER_FRAMES)
        self._render_status(self._snapshot)

    def _render_status(self, snapshot: StatusSnapshot) -> None:
        if snapshot.running:
            icon = _SPINNER_FRAMES[self._frame]
            state_text = "researching"
        elif snapshot.reason:
            icon = "✅" if "complete" in snapshot.reason else "▪"
            state_text = snapshot.reason
        else:
            icon = "💤"
            state_text = "idle"

        parts = [
            f"{icon} [b]{state_text}[/]",
            f"iteration: {snapshot.iteration}/{snapshot.max_iterations}",
            f"findings: {snapshot.findings}",
        ]
        self.update("   ".join(parts))


# ── Title ─────────────────────────────────────────────────────────────────────


class Title(Static):
    """Static heading."""

    DEFAULT_CSS = "Title { text-style: bold; height: 1; }"


class ReportList(ListView):
    """List of persisted reports, newest first.

    Populate it after mount with :meth:`set_reports`; ``ListView.append``
    cannot run before the widget is mounted.
    """

    DEFAULT_CSS = """
    ReportList > ListItem {
        padding: 0 1;
        height: 1;
    }
    ReportList > ListItem.--highlight {
        border-left: thick $accent;
        background: $boost;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reports_map: dict[ListItem, Report] = {}

    async def set_reports(self, reports: Sequence[Report]) -> None:
        """Replace the displayed reports with *reports*."""
        self.reports_map.clear()
        await super().clear()
        for report in reports:
            item = ListItem(
                Label(escape(report.topic[:80])),
                classes="report-item",
            )
            self.reports_map[item] = report
            await self.append(item)

    def clear(self) -> None:  # type: ignore[override]
        self.reports_map.clear()
        return super().clear()  # type: ignore[return-value]


# ── ExportMenu ────────────────────────────────────────────────────────────────

#: Friendly group headings, keyed by the ``category`` an exporter declares.
_CATEGORY_LABELS: dict[str, str] = {
    "local_file": "Save to file",
    "ai_chat": "Open in AI chat",
}

#: Menu entry captions, keyed by export format id.
_FORMAT_LABELS: dict[str, str] = {
    "md": "📄  Markdown",
    "pdf": "📕  PDF",
    "html": "🌐  HTML",
    "chatgpt": "💬  ChatGPT",
    "claude": "✴️  Claude",
}

#: Pseudo-format of the "save every local file at once" entry.
_BULK_FORMAT = "all"

_TRIGGER_LABEL = "⇩ Export ▾"
_TRIGGER_LABEL_OPEN = "⇩ Export ▴"
_TRIGGER_LABEL_BUSY = "⏳ Exporting…"


def _category_label(category: str) -> str:
    """Human heading for an exporter category, falling back to its raw name."""
    return _CATEGORY_LABELS.get(category, category.replace("_", " ").title())


def _format_label(category: str, fmt: str) -> str:
    """Caption for a single menu entry."""
    label = _FORMAT_LABELS.get(fmt)
    if label is not None:
        return label
    return f"{fmt}  .{fmt}" if category == "local_file" else fmt.upper()


class ExportMenuOverlay(OptionList):
    """Dropdown panel listing every available export format.

    Owned by :class:`ExportMenu`; it only reports intent back to it.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss", "Close", show=False),
    ]

    class Dismiss(Message):
        """The overlay should close, either via escape or by losing focus."""

        def __init__(self, lost_focus: bool = False) -> None:
            super().__init__()
            self.lost_focus = lost_focus

    class Selected(Message):
        """An option with *option_id* was activated."""

        def __init__(self, option_id: str) -> None:
            super().__init__()
            self.option_id = option_id

    def action_dismiss(self) -> None:
        self.post_message(self.Dismiss())

    def _on_blur(self, event: events.Blur) -> None:
        # Only an *open* menu reacts to losing focus, and a press on the trigger
        # is left to the trigger's own click: a dismiss queued by that press
        # would otherwise race the click and leave the menu in the wrong state.
        parent = self.parent
        if not isinstance(parent, ExportMenu) or not parent.expanded:
            return
        if parent.pointer_on_trigger:
            return
        self.post_message(self.Dismiss(lost_focus=True))

    def _on_resize(self) -> None:
        # The panel is right-aligned against the trigger, so it has to be
        # re-aligned whenever the panel itself is re-measured.
        parent = self.parent
        if isinstance(parent, ExportMenu):
            parent._align_overlay()

    @on(OptionList.OptionSelected)
    def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        option = self.get_option_at_index(event.option_index)
        if option.id is not None:
            self.post_message(self.Selected(option.id))


class _ExportTrigger(Static):
    """Flat pill that opens the dropdown; all logic lives in ExportMenu."""

    class Toggle(Message):
        """The trigger was clicked."""

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.post_message(self.Toggle())


class ExportMenu(Horizontal, can_focus=True):
    """Export button that expands into a dropdown of export formats.

    Feed it the output of ``ExportService.get_available_formats()`` with
    :meth:`set_formats`; activating an entry posts :class:`ExportMenu.ExportRequested`
    carrying the format ids to export. Disabled until both formats and a report
    are available.
    """

    DEFAULT_CSS = """
    ExportMenu {
        width: auto;
        height: 1;

        & > .export-trigger {
            width: auto;
            min-width: 16;
            height: 1;
            padding: 0 1;
            color: $accent;
            background: $accent 12%;
            text-align: center;
            text-style: bold;
        }

        &:focus > .export-trigger,
        & > .export-trigger:hover {
            background: $accent 28%;
        }

        &.-expanded > .export-trigger {
            background: $accent 40%;
        }

        &.-busy > .export-trigger {
            color: $text-muted;
            background: transparent;
        }

        &:disabled > .export-trigger {
            color: $text-disabled;
            background: transparent;
        }

        & > .export-overlay {
            display: none;
            overlay: screen;
            width: 40;
            height: auto;
            max-height: 14;
            background: $panel;
            border: round $accent;
        }

        &.-expanded > .export-overlay {
            display: block;
        }
    }
    ExportMenu .export-overlay:focus {
        background-tint: $foreground 3%;
    }
    ExportMenu .export-overlay > .option-list--option {
        padding: 0 1;
    }
    ExportMenu .export-overlay > .option-list--option-highlighted {
        background: $accent 30%;
        color: $text;
        text-style: bold;
    }
    ExportMenu .export-overlay > .option-list--option-disabled {
        color: $text-muted;
        text-style: bold;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter,space,down", "toggle_menu", "Export", show=False),
    ]

    expanded: var[bool] = var(False, init=False)
    """True while the dropdown is open."""

    busy: var[bool] = var(False, init=False)
    """True while an export triggered from this menu is in flight."""

    def __init__(
        self,
        categories: Mapping[str, Sequence[str]] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._formats: dict[str, tuple[str, ...]] = {}
        self._has_formats = False
        self._has_report = False
        # Categories are applied in on_mount: the overlay only exists after compose().
        self._pending_categories = categories

    def compose(self) -> ComposeResult:
        yield _ExportTrigger(_TRIGGER_LABEL, classes="export-trigger")
        yield ExportMenuOverlay(classes="export-overlay")

    def on_mount(self) -> None:
        self.query_one(_ExportTrigger).tooltip = "Export this report  (ctrl+e)"
        if self._pending_categories:
            self.set_formats(self._pending_categories)
            self._pending_categories = None
        self._sync_enabled()

    def validate_expanded(self, expanded: bool) -> bool:
        return expanded and not self.disabled

    # ── public API ──────────────────────────────────────────────────────

    def set_formats(self, categories: Mapping[str, Sequence[str]]) -> None:
        """Rebuild the dropdown from an ``ExportService`` category mapping."""
        overlay = self.query_one(ExportMenuOverlay)
        overlay.clear_options()
        self._formats = {}

        for category, formats in categories.items():
            formats = tuple(formats)
            if not formats:
                continue
            if self._formats:
                overlay.add_option(None)
            overlay.add_option(
                Option(
                    Content.from_markup(f"[_category_]{_category_label(category)}[/]"),
                    disabled=True,
                )
            )
            for fmt in formats:
                overlay.add_option(Option(_format_label(category, fmt), id=fmt))
                self._formats[fmt] = (fmt,)

        local_formats = tuple(categories.get("local_file", ()))
        if len(local_formats) > 1:
            extensions = " · ".join(f".{fmt}" for fmt in local_formats)
            overlay.add_option(None)
            overlay.add_option(
                Option(
                    Content.from_markup(f"⬇  Save all   [_dim_]{extensions}[/]"),
                    id=_BULK_FORMAT,
                )
            )
            self._formats[_BULK_FORMAT] = local_formats

        self._has_formats = bool(self._formats)
        self._sync_enabled()

    def set_report_available(self, available: bool) -> None:
        """Enable or disable the menu based on whether a report is on screen."""
        self._has_report = available
        self._sync_enabled()

    @property
    def pointer_on_trigger(self) -> bool:
        """True while the mouse hovers the button, so a press there is its own."""
        trigger = self.query_one(_ExportTrigger)
        return trigger.region.contains_point(self.app.mouse_position)

    # ── state ───────────────────────────────────────────────────────────

    def _sync_enabled(self) -> None:
        self.disabled = not (self._has_formats and self._has_report)

    def _watch_expanded(self, expanded: bool) -> None:
        self.set_class(expanded, "-expanded")
        self.query_one(_ExportTrigger).update(
            _TRIGGER_LABEL_OPEN if expanded else _TRIGGER_LABEL
        )
        overlay = self.query_one(ExportMenuOverlay)
        if not expanded:
            return
        overlay.focus(scroll_visible=False)
        if overlay.highlighted is None:
            overlay.action_first()
        self.call_after_refresh(self._align_overlay)

    def _on_resize(self, _event: events.Resize) -> None:
        self._align_overlay()

    def _watch_busy(self, busy: bool) -> None:
        self.set_class(busy, "-busy")
        trigger = self.query_one(_ExportTrigger)
        if busy:
            trigger.update(_TRIGGER_LABEL_BUSY)
        else:
            trigger.update(_TRIGGER_LABEL_OPEN if self.expanded else _TRIGGER_LABEL)

    def _align_overlay(self) -> None:
        """Hang the dropdown off the right edge of the trigger.

        A ``Horizontal`` parent places an ``overlay: screen`` child at the flow
        position it would otherwise occupy -- here, directly after the trigger --
        so the panel only has to grow leftwards to sit flush with the button and
        clear of the screen edge.  ``outer_size`` is measured rather than
        ``size`` because the panel carries a border.
        """
        overlay = self.query_one(ExportMenuOverlay)
        overlay.styles.offset = (-overlay.outer_size.width, self.size.height)

    # ── actions ─────────────────────────────────────────────────────────

    def action_toggle_menu(self) -> None:
        if self.busy or self.disabled:
            return
        self.expanded = not self.expanded

    @on(_ExportTrigger.Toggle)
    def _on_trigger_toggled(self, _event: _ExportTrigger.Toggle) -> None:
        self.action_toggle_menu()

    @on(ExportMenuOverlay.Dismiss)
    def _on_overlay_dismissed(self, event: ExportMenuOverlay.Dismiss) -> None:
        event.stop()
        self.expanded = False
        if not event.lost_focus:
            self.focus()

    @on(ExportMenuOverlay.Selected)
    def _on_overlay_selected(self, event: ExportMenuOverlay.Selected) -> None:
        event.stop()
        formats = self._formats.get(event.option_id)
        self.expanded = False
        self.focus()
        if formats:
            self.post_message(self.ExportRequested(formats))

    class ExportRequested(Message):
        """The user picked *formats* (ids) to export the current report to."""

        def __init__(self, formats: tuple[str, ...]) -> None:
            super().__init__()
            self.formats = formats

        def __rich_repr__(self) -> Generator[tuple[str, ...]]:
            yield self.formats


# ── ReportDetailPanel ─────────────────────────────────────────────────────────

_PLACEHOLDER = "_Select a report to read it._"

_EXPORT_STATUS_TIMEOUT = 8.0


class ReportDetailPanel(Vertical):
    """Right-hand panel of the master-detail Reports layout.

    Composes a top bar (toggle button, title, export status and the export
    dropdown) above a scrollable Markdown pane.  Call :meth:`show` to populate
    it with a report; :meth:`clear` to reset it to the placeholder state.
    """

    DEFAULT_CSS = """
    ReportDetailPanel {
        width: 1fr;
        height: 1fr;
    }
    ReportDetailPanel #report-topbar {
        height: 3;
        border-bottom: solid $border;
        align-vertical: middle;
    }
    ReportDetailPanel #toggle-sidebar {
        width: 5;
        min-width: 5;
        background: transparent;
        border: none;
        color: $text-muted;
    }
    ReportDetailPanel #toggle-sidebar:hover {
        background: $boost;
        color: $text;
        border: none;
    }
    ReportDetailPanel #report-title {
        width: 1fr;
        content-align: left middle;
        height: 3;
        padding: 0 1;
        color: $text-muted;
    }
    ReportDetailPanel #export-status {
        display: none;
        width: auto;
        max-width: 40;
        height: 1;
        content-align: right middle;
        color: $text-muted;
        text-style: none;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    ReportDetailPanel #export-menu {
        margin-right: 1;
    }
    ReportDetailPanel #report-content {
        height: 1fr;
        overflow-y: auto;
        scrollbar-size-vertical: 1;
        padding: 0 2;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_report: Report | None = None
        self._status_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="report-topbar"):
            yield Button("≡", id="toggle-sidebar")
            yield Static("", id="report-title")
            yield Static("", id="export-status")
            yield ExportMenu(id="export-menu")
        with VerticalScroll(id="report-content-scroll"):
            yield Markdown(_PLACEHOLDER, id="report-content")

    @property
    def current_report(self) -> Report | None:
        """The report on screen, or ``None`` while showing the placeholder."""
        return self._current_report

    def set_export_formats(self, categories: Mapping[str, Sequence[str]]) -> None:
        """Give the export dropdown the formats to offer."""
        self.query_one("#export-menu", ExportMenu).set_formats(categories)

    def show_export_status(self, message: str) -> None:
        """Show *message* in the top bar until :data:`_EXPORT_STATUS_TIMEOUT`."""
        status = self.query_one("#export-status", Static)
        status.update(message)
        status.display = True
        if self._status_timer is not None:
            self._status_timer.stop()
        self._status_timer = self.set_timer(
            _EXPORT_STATUS_TIMEOUT, self.clear_export_status
        )

    def clear_export_status(self) -> None:
        """Hide the top-bar export status."""
        if self._status_timer is not None:
            self._status_timer.stop()
            self._status_timer = None
        status = self.query_one("#export-status", Static)
        status.display = False
        status.update("")

    def show(self, report: Report) -> None:
        """Populate the panel with *report*."""
        self._current_report = report
        self.query_one("#export-menu", ExportMenu).set_report_available(True)
        self.query_one("#report-title", Static).update(escape(report.topic))
        md = self.query_one("#report-content", Markdown)
        md.update(report.content)
        md.scroll_home(animate=False)

    def clear(self) -> None:
        """Reset the panel to its placeholder state."""
        self._current_report = None
        self.clear_export_status()
        self.query_one("#export-menu", ExportMenu).set_report_available(False)
        self.query_one("#report-title", Static).update("")
        self.query_one("#report-content", Markdown).update(_PLACEHOLDER)


__all__ = [
    "CollapsibleSettings",
    "ExportMenu",
    "ExportMenuOverlay",
    "FindingDetailPane",
    "FindingsList",
    "ProgressFeed",
    "ReportDetailPanel",
    "ReportList",
    "StatusBar",
    "StatusSnapshot",
    "Title",
]
