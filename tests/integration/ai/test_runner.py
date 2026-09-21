import asyncio
import os
import unittest

from deep_research_agent.ai.agent.runner import AgentRunner
from deep_research_agent.ai.llm.openrouter import OpenRouterLLMProvider
from deep_research_agent.config import load_environment
from deep_research_agent.services.search.service import SearchService
from deep_research_agent.services.search.tavily import TavilySearchProvider


class TestAgentRunnerIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        load_environment()
        if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("TAVILY_API_KEY"):
            raise unittest.SkipTest(
                "OPENROUTER_API_KEY / TAVILY_API_KEY environment variables not set"
            )
        llm = OpenRouterLLMProvider(
            model_name="nex-agi/nex-n2.5-mini:free",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        search_service = SearchService(provider=TavilySearchProvider())
        self.runner = AgentRunner(
            search_service=search_service, llm=llm, max_iterations=1
        )

    async def test_live_agent_run_produces_report(self):
        state = await asyncio.wait_for(
            self.runner.run("What is a vector database?"), timeout=180
        )

        self.assertTrue(state.is_complete)
        self.assertLessEqual(state.current_iteration, 1)
        self.assertGreater(len(state.findings), 0)
        report = state.synthesized_report
        self.assertIsInstance(report, str)
        assert report is not None
        self.assertTrue(report.strip())
        self.assertNotIn(
            "No findings were gathered during this research session.", report
        )


if __name__ == "__main__":
    unittest.main()
