import unittest

from deep_research_agent.services.search.tavily import TavilySearchProvider

from .base import BaseProviderIntegrationTest


class TestTavilyIntegration(BaseProviderIntegrationTest):
    api_key_env = "TAVILY_API_KEY"

    def _create_provider(self, api_key: str) -> TavilySearchProvider:
        return TavilySearchProvider(api_key=api_key)


if __name__ == "__main__":
    unittest.main()
