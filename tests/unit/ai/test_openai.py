import unittest
from types import SimpleNamespace

from openai import BadRequestError

from deep_research_agent.ai.llm.base import LLMRequest
from deep_research_agent.ai.llm.openai import OpenAILLMProvider
from tests.unit.ai.base import Answer, BaseProviderUnitTest, _fake_response


def _structured_response():
    return _fake_response(content='{"answer": "42", "confidence": 0.9}')


def _bad_request() -> BadRequestError:
    response = SimpleNamespace(
        request=SimpleNamespace(),
        status_code=400,
        headers={},
    )
    return BadRequestError(
        "response_format json_schema not supported",
        response=response,  # type: ignore[arg-type]
        body=None,
    )


class TestOpenAILLMProvider(BaseProviderUnitTest):
    def _create_provider(self) -> OpenAILLMProvider:
        return OpenAILLMProvider(model_name="gpt-4o-mini", api_key="test-key")

    async def test_token_usage_mapped(self):
        self.create.return_value = _fake_response(prompt_tokens=10, completion_tokens=5)

        response = await self.provider.generate(LLMRequest(prompt="hello"))

        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 5)

    async def test_system_and_user_messages_built(self):
        await self.provider.generate(
            LLMRequest(prompt="hello", system_prompt="be terse")
        )

        messages = self.create.call_args.kwargs["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "be terse"})
        self.assertEqual(messages[1], {"role": "user", "content": "hello"})

    async def test_temperature_passed_through(self):
        await self.provider.generate(LLMRequest(prompt="hello", temperature=0.2))

        self.assertEqual(self.create.call_args.kwargs["temperature"], 0.2)

    async def test_structured_output_uses_json_schema(self):
        self.create.return_value = _structured_response()

        response = await self.provider.generate(
            LLMRequest(prompt="What is 6*7?", response_model=Answer)
        )

        self.assertIsInstance(response.content, Answer)
        self.assertEqual(response.content.answer, "42")
        self.assertEqual(response.content.confidence, 0.9)

        response_format = self.create.call_args.kwargs["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "Answer")
        self.assertIn("schema", response_format["json_schema"])

    async def test_structured_output_invalid_json_raises(self):
        self.create.return_value = _fake_response(content="not json")

        with self.assertRaises(ValueError):
            await self.provider.generate(LLMRequest(prompt="x", response_model=Answer))

    async def test_json_schema_rejected_falls_back_to_json_object(self):
        self.create.side_effect = [_bad_request(), _structured_response()]

        response = await self.provider.generate(
            LLMRequest(prompt="x", response_model=Answer)
        )

        self.assertIsInstance(response.content, Answer)
        first_format = self.create.call_args_list[0].kwargs["response_format"]
        second_format = self.create.call_args_list[1].kwargs["response_format"]
        self.assertEqual(first_format["type"], "json_schema")
        self.assertEqual(second_format["type"], "json_object")


if __name__ == "__main__":
    unittest.main()
