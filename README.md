# Deep Research Agent

A stateful, autonomous deep research engine built with Python, LangGraph, PostgreSQL (`pgvector`), and Playwright. Designed to execute long-horizon web investigation, maintain dual-layer memory, resolve knowledge conflicts, and generate cited markdown reports.

## System Architecture

The project is structured into four decoupled modules:
1. **Orchestrator:** LangGraph state machine handling iteration loops, circuit breakers, and state checkpoints.
2. **Search Service:** A web search service with support for multiple providers.
3. **Memory Service** Support for short and long term memory.
 
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
