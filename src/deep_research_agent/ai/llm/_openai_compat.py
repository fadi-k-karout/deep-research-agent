import json
import logging
import re
from typing import cast

from openai import AsyncOpenAI, BadRequestError
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import ValidationError

from .base import BaseLLMProvider, LLMRequest, LLMResponse, T

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _extract_json(raw: str) -> str | None:
    """Extract the outermost JSON object from ``raw``.

    Providers may wrap structured output in markdown fences or add
    surrounding prose; this strips fences and grabs the first ``{...}``
    span so it can still be validated as JSON.
    """
    start = raw.find("{")
    if start == -1:
        return None
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fenced is not None:
        return fenced.group(1)
    end = raw.rfind("}")
    if end < start:
        return None
    return raw[start : end + 1]


class BaseOpenAICompatibleProvider(BaseLLMProvider):
    """Shared adapter for OpenAI-compatible chat completions APIs.

    Used by the OpenAI and OpenRouter providers, which differ only in the
    API base URL, the API key, and whether the ``json_schema`` structured
    output mode is enabled.
    """

    use_json_schema: bool = True

    def __init__(
        self,
        model_name: str,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model_name=model_name, api_key=api_key)
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def generate(self, request: LLMRequest[T]) -> LLMResponse[T]:
        """Generates a completion and returns the parsed response.

        When ``request.response_model`` is provided the content is parsed
        into that pydantic model. Providers with ``use_json_schema`` request
        OpenAI's ``json_schema`` response format and fall back to plain
        ``json_object`` followed by local validation if the model rejects it.
        """
        response_format: dict[str, object] | None = None
        if request.response_model is not None:
            if self.use_json_schema:
                response_format = self._json_schema_format(request.response_model)
            else:
                response_format = {"type": "json_object"}
            user_prompt = self._prompt_with_schema(
                request.prompt, request.response_model
            )
        else:
            user_prompt = request.prompt

        messages = self._build_messages(request.system_prompt, user_prompt)

        try:
            response = await self._complete(
                messages, request.temperature, response_format
            )
        except BadRequestError:
            if request.response_model is None or not self.use_json_schema:
                raise
            logger.warning(
                "model %r rejected response_format json_schema; retrying as json_object",
                self.model_name,
            )
            response = await self._complete(
                messages,
                request.temperature,
                {"type": "json_object"},
            )

        content = self._extract_content(response)
        usage = response.usage
        kwargs = {
            "raw_response": content,
            "model_name": response.model or self.model_name,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
        }

        if request.response_model is not None:
            parsed = self._parse_structured(content, request.response_model)
            return cast(LLMResponse[T], LLMResponse(content=parsed, **kwargs))

        return cast(LLMResponse[T], LLMResponse(content=content, **kwargs))

    async def _complete(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float,
        response_format: dict[str, object] | None,
    ) -> ChatCompletion:
        if response_format is None:
            return await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )
        return await self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            response_format=cast(ResponseFormat, response_format),
        )

    @staticmethod
    def _build_messages(
        system_prompt: str | None, user_prompt: str
    ) -> list[ChatCompletionMessageParam]:
        return [
            {"role": "system", "content": system_prompt or _DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _json_schema_format(model: type[T]) -> dict[str, object]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "strict": True,
                "schema": model.model_json_schema(),
            },
        }

    @staticmethod
    def _prompt_with_schema(prompt: str, model: type[T]) -> str:
        schema = json.dumps(model.model_json_schema())
        return (
            f"{prompt}\n\n"
            "Respond with a single valid JSON object — no markdown, no code "
            "fences, no commentary — matching this JSON schema:\n"
            f"{schema}"
        )

    @staticmethod
    def _extract_content(response: ChatCompletion) -> str:
        if not response.choices:
            raise ValueError("Received a completion with no choices")
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Received an empty completion")
        return content

    def _parse_structured(self, raw: str, model: type[T]) -> T:
        try:
            return model.model_validate_json(raw)
        except ValidationError as direct_error:
            candidate = _extract_json(raw)
            if candidate is None or candidate == raw:
                raise ValueError(
                    f"LLM returned content that does not conform to {model.__name__}: "
                    f"{direct_error}"
                ) from direct_error
            try:
                return model.model_validate_json(candidate)
            except ValidationError as extracted_error:
                raise ValueError(
                    f"LLM returned content that does not conform to {model.__name__}: "
                    f"{extracted_error}"
                ) from extracted_error
