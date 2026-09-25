# Deep Research Agent

A stateful, autonomous deep research engine that iterates on web searches, extracts findings, and synthesizes cited Markdown reports. Reports are persisted to a local SQLite database and can be reviewed later from the TUI.

## System Architecture

The project is structured into four decoupled modules:
1. **Orchestrator:** The `AgentRunner` iteration loop handling planning, search, extraction, synthesis, and termination reasons.
2. **Search Service:** A web search service with support for multiple providers (Tavily, Exa).
3. **Storage Service:** `ResearchAgentStorage` persists each finished report to a local SQLite database.
4. **TUI:** A Textual interface with a Progress tab, a Report tab, and a Reports tab for browsing stored reports.

## Report Storage

Every completed run is saved to `deep_research_agent.db` (SQLite) in the working directory. The database file is created on first use and is excluded from version control.

One storage instance is created per application session and passed explicitly to the TUI, the CLI, and the runner — there is no global database connection:

```
main() -> run_tui() -> DeepResearchApp -> build_runner() -> AgentRunner
```

The TUI loads stored reports when it starts and refreshes the list after each run. Select a report in the **Reports** tab (`ctrl+3`) to open it in the **Report** tab.

## Getting Started

### Prerequisites
- [uv](https://astral.sh/uv/) package manager
- The required python version is in the pyproject.toml

### Environment Setup
```bash
# Clone and enter the repository
git clone <repo-url>
cd deep-research-agent

# Sync dependencies with uv
uv sync
```

## Development

Common commands are defined in the [Taskfile.yml](Taskfile.yml) and run via [go-task](https://taskfile.dev/):

```bash
# Show all available tasks
task

# Run unit tests
task test:unit

# Run integration tests (skips without API keys)
task test:integration

# Run all tests
task test

# Run the CLI
task run

# Sync dependencies
task sync
```

### Running Tests
Tests use Python's built-in `unittest`:

```bash
uv run python -m unittest discover -s tests/unit -v        # unit tests
uv run python -m unittest discover -s tests/integration -v # integration tests (need TAVILY_API_KEY/EXA_API_KEY)
```
