import unittest

from deep_research_agent.services.search.exa import ExaSearchProvider

from .base import BaseProviderIntegrationTest


class TestExaIntegration(BaseProviderIntegrationTest):
    api_key_env = "EXA_API_KEY"

    def _create_provider(self, api_key: str) -> ExaSearchProvider:
        return ExaSearchProvider(api_key=api_key)


if __name__ == "__main__":
    unittest.main()
