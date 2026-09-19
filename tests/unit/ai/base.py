import unittest
from abc import ABC, abstractmethod
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

from deep_research_agent.ai.llm.base import BaseLLMProvider, LLMRequest


class Answer(BaseModel):
    answer: str
    confidence: float


def _fake_response(
    content: str = "hi",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 12,
    completion_tokens: int = 4,
) -> SimpleNamespace:
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    return SimpleNamespace(
        choices=[choice],
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


class BaseProviderUnitTest(unittest.IsolatedAsyncioTestCase, ABC):
    @abstractmethod
    def _create_provider(self) -> BaseLLMProvider:
        """Instantiate the provider under test."""

    def setUp(self):
        self.provider = self._create_provider()
        self.create = AsyncMock(
            return_value=_fake_response(model=self.provider.model_name)
        )
        patcher = patch.object(
            self.provider._client.chat.completions, "create", self.create
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_plain_generation_returns_string(self):
        response = await self.provider.generate(LLMRequest(prompt="hello"))

        self.assertIsInstance(response.content, str)
        self.assertEqual(response.content, "hi")
        self.assertTrue(response.raw_response)
        self.assertEqual(response.model_name, self.provider.model_name)
        self.assertNotIn("response_format", self.create.call_args.kwargs)

    async def test_empty_choices_raises(self):
        self.create.return_value = _fake_response(model=self.provider.model_name)
        self.create.return_value.choices = []

        with self.assertRaises(ValueError):
            await self.provider.generate(LLMRequest(prompt="hello"))
