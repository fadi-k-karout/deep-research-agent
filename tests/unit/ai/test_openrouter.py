import unittest

from deep_research_agent.ai.llm.base import LLMRequest
from deep_research_agent.ai.llm.openrouter import OpenRouterLLMProvider
from tests.unit.ai.base import Answer, BaseProviderUnitTest, _fake_response


class TestOpenRouterLLMProvider(BaseProviderUnitTest):
    def _create_provider(self) -> OpenRouterLLMProvider:
        return OpenRouterLLMProvider(
            model_name="openai/gpt-4o-mini", api_key="test-key"
        )

    async def test_structured_output_uses_json_object(self):
        self.create.return_value = _fake_response(
            content='{"answer": "Paris", "confidence": 0.95}'
        )

        response = await self.provider.generate(
            LLMRequest(prompt="Capital of France?", response_model=Answer)
        )

        self.assertIsInstance(response.content, Answer)
        self.assertEqual(response.content.answer, "Paris")
        response_format = self.create.call_args.kwargs["response_format"]
        self.assertEqual(response_format, {"type": "json_object"})

    async def test_structured_output_injects_schema_hint(self):
        self.create.return_value = _fake_response(
            content='{"answer": "Paris", "confidence": 0.9}'
        )

        await self.provider.generate(
            LLMRequest(prompt="Capital of France?", response_model=Answer)
        )

        messages = self.create.call_args.kwargs["messages"]
        self.assertIn("JSON", messages[-1]["content"])
        self.assertIn("schema", messages[-1]["content"].lower())

    async def test_markdown_fenced_json_is_parsed(self):
        self.create.return_value = _fake_response(
            content='```json\n{"answer": "Paris", "confidence": 0.9}\n```'
        )

        response = await self.provider.generate(
            LLMRequest(prompt="Capital of France?", response_model=Answer)
        )

        self.assertIsInstance(response.content, Answer)
        self.assertEqual(response.content.answer, "Paris")

    async def test_prose_wrapped_json_is_parsed(self):
        self.create.return_value = _fake_response(
            content='Sure! Here is the answer: {"answer": "Rome", "confidence": 0.8}'
        )

        response = await self.provider.generate(
            LLMRequest(prompt="x", response_model=Answer)
        )

        self.assertIsInstance(response.content, Answer)
        self.assertEqual(response.content.answer, "Rome")

    async def test_invalid_json_raises(self):
        self.create.return_value = _fake_response(content="sorry, no json here")

        with self.assertRaises(ValueError):
            await self.provider.generate(LLMRequest(prompt="x", response_model=Answer))


if __name__ == "__main__":
    unittest.main()
