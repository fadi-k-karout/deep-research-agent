import argparse
import asyncio

from deep_research_agent.ai.agent import AgentRunner
from deep_research_agent.ai.llm.openrouter import OpenRouterLLMProvider
from deep_research_agent.config import load_environment
from deep_research_agent.services.search.service import SearchService
from deep_research_agent.services.search.tavily import TavilySearchProvider

_DEFAULT_MODEL = "openrouter/auto"


def _build_runner(max_iterations: int) -> AgentRunner:
    llm = OpenRouterLLMProvider(model_name="nex-agi/nex-n2.5-mini:free")
    search_service = SearchService(provider=TavilySearchProvider())
    return AgentRunner(
        search_service=search_service,
        llm=llm,
        max_iterations=max_iterations,
    )


async def _run(prompt: str, max_iterations: int) -> str:
    runner = _build_runner(max_iterations)
    state = await runner.run(prompt)
    return state.synthesized_report or ""


def main() -> None:
    load_environment()

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
    args = parser.parse_args()

    prompt = args.prompt
    if not prompt:
        prompt = input("Research prompt: ")

    report = asyncio.run(_run(prompt, args.max_iterations))
    print(report)
