"""Custom widgets for the deep research TUI."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    RichLog,
    Select,
    Static,
)

from deep_research_agent.ai.state import Finding

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


__all__ = [
    "CollapsibleSettings",
    "FindingDetailPane",
    "FindingsList",
    "ProgressFeed",
    "StatusBar",
    "StatusSnapshot",
    "Title",
]
