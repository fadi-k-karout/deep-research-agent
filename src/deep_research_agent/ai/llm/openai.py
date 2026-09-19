from deep_research_agent.config import openai_api_key

from ._openai_compat import BaseOpenAICompatibleProvider

OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAILLMProvider(BaseOpenAICompatibleProvider):
    """LLM provider backed by the OpenAI chat completions API."""

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
            api_key=api_key or openai_api_key(),
            base_url=OPENAI_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
        )
