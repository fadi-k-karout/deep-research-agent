from deep_research_agent.config import openrouter_api_key

from ._openai_compat import BaseOpenAICompatibleProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLLMProvider(BaseOpenAICompatibleProvider):
    """LLM provider backed by OpenRouter's OpenAI-compatible API.

    Uses ``json_object`` rather than ``json_schema`` because not all
    OpenRouter models support OpenAI's structured output mode; contents are
    validated locally against the requested pydantic model instead.
    """

    use_json_schema = False

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_key=api_key or openrouter_api_key(),
            base_url=OPENROUTER_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
