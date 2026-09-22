import argparse
import asyncio

from deep_research_agent.ai.agent import AgentRunner
from deep_research_agent.ai.llm.openrouter import OpenRouterLLMProvider
from deep_research_agent.config import load_environment
from deep_research_agent.events import EventEmitter
from deep_research_agent.services.search.base import BaseSearchProvider
from deep_research_agent.services.search.exa import ExaSearchProvider
from deep_research_agent.services.search.service import SearchService
from deep_research_agent.services.search.tavily import TavilySearchProvider

DEFAULT_MODEL = "nex-agi/nex-n2.5-mini:free"
SEARCH_PROVIDERS = {"tavily": TavilySearchProvider, "exa": ExaSearchProvider}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-research-agent",
        description="Run a deep research session and print a cited markdown report.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Research objective. Prompts interactively when omitted.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of search/planning loops (default: 5).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--search-provider",
        choices=sorted(SEARCH_PROVIDERS),
        default="tavily",
        help="Search provider to use (default: tavily).",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch the interactive terminal UI instead of headless mode.",
    )
    return parser


def build_search_provider(name: str) -> BaseSearchProvider:
    try:
        provider_cls = SEARCH_PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown search provider: {name!r} "
            f"(expected one of {sorted(SEARCH_PROVIDERS)})"
        ) from None
    return provider_cls()


def build_runner(
    max_iterations: int,
    model: str = DEFAULT_MODEL,
    search_provider: str = "tavily",
    events: EventEmitter | None = None,
) -> AgentRunner:
    llm = OpenRouterLLMProvider(model_name=model)
    search_service = SearchService(provider=build_search_provider(search_provider))
    return AgentRunner(
        search_service=search_service,
        llm=llm,
        max_iterations=max_iterations,
        events=events,
    )


async def _run(
    prompt: str,
    max_iterations: int,
    model: str = DEFAULT_MODEL,
    search_provider: str = "tavily",
) -> str:
    runner = build_runner(
        max_iterations=max_iterations,
        model=model,
        search_provider=search_provider,
    )
    state = await runner.run(prompt)
    return state.synthesized_report or ""


def main(argv: list[str] | None = None) -> None:
    load_environment()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_iterations <= 0:
        parser.error("--max-iterations must be a positive integer")

    if args.tui:
        from deep_research_agent.tui.app import run_tui

        run_tui(
            prompt=args.prompt,
            max_iterations=args.max_iterations,
            model=args.model,
            search_provider=args.search_provider,
        )
        return

    prompt = args.prompt
    if not prompt:
        prompt = input("Research prompt: ")

    report = asyncio.run(
        _run(
            prompt,
            args.max_iterations,
            model=args.model,
            search_provider=args.search_provider,
        )
    )
    print(report)
