import os
import unittest
from abc import ABC, abstractmethod

from pydantic import BaseModel

from deep_research_agent.ai.llm.base import BaseLLMProvider, LLMRequest
from deep_research_agent.config import load_environment


class _Answer(BaseModel):
    answer: str


class BaseLLMIntegrationTest(unittest.IsolatedAsyncioTestCase, ABC):
    api_key_env: str | None = None

    @abstractmethod
    def _create_provider(self, api_key: str) -> BaseLLMProvider:
        """Instantiate the provider under test with the given API key."""

    async def asyncSetUp(self):
        load_environment()
        if not self.api_key_env:
            raise unittest.SkipTest("api_key_env not configured on test class")
        self.api_key = os.getenv(self.api_key_env)
        if not self.api_key:
            raise unittest.SkipTest(f"{self.api_key_env} environment variable not set")
        self.provider = self._create_provider(self.api_key)

    async def test_live_generation_returns_string(self):
        response = await self.provider.generate(
            LLMRequest(prompt="Reply with the single word: pong.")
        )

        self.assertIsInstance(response.content, str)
        self.assertTrue(response.content)
        self.assertTrue(response.raw_response)
        self.assertTrue(response.model_name)

    async def test_live_structured_output_parses(self):
        response = await self.provider.generate(
            LLMRequest(
                prompt="In one word, name the capital of France.",
                response_model=_Answer,
            )
        )

        self.assertIsInstance(response.content, _Answer)
        self.assertTrue(response.content.answer)
