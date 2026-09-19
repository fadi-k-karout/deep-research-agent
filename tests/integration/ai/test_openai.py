import unittest

from deep_research_agent.ai.llm.openai import OpenAILLMProvider
from tests.integration.ai.base import BaseLLMIntegrationTest


class TestOpenAILLMIntegration(BaseLLMIntegrationTest):
    api_key_env = "OPENAI_API_KEY"

    def _create_provider(self, api_key: str) -> OpenAILLMProvider:
        return OpenAILLMProvider(model_name="gpt-4o-mini", api_key=api_key)


if __name__ == "__main__":
    unittest.main()
