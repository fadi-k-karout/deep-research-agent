import unittest

from deep_research_agent.ai.llm.openrouter import OpenRouterLLMProvider
from tests.integration.ai.base import BaseLLMIntegrationTest


class TestOpenRouterLLMIntegration(BaseLLMIntegrationTest):
    api_key_env = "OPENROUTER_API_KEY"

    def _create_provider(self, api_key: str) -> OpenRouterLLMProvider:
        return OpenRouterLLMProvider(model_name="openai/gpt-oss-20b", api_key=api_key)


if __name__ == "__main__":
    unittest.main()
