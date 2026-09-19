from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, Field

# Generic Type variable for structured response payloads
T = TypeVar("T", bound=BaseModel)


class LLMRequest[T: BaseModel](BaseModel):
    """Standardized request payload for LLM invocations."""

    prompt: str = Field(description="User prompt or main context input.")
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt to guide LLM behavior."
    )
    temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for response variance.",
    )
    response_model: type[T] | None = Field(
        default=None,
        description="Optional Pydantic model type to enforce structured JSON output.",
    )


class LLMResponse[R](BaseModel):
    """Standardized response wrapper containing parsed content and token usage.

    ``R`` is the static content payload type: the requested pydantic model
    when ``response_model`` was provided, otherwise ``str`` for the raw
    plain-text response.
    """

    content: R = Field(
        description="Parsed Pydantic model instance if response_model was provided, else raw string."
    )
    raw_response: str = Field(
        description="Raw string response returned by the underlying LLM provider."
    )
    model_name: str = Field(
        description="Name of the model that generated this response."
    )
    prompt_tokens: int = Field(
        default=0, description="Tokens consumed by prompt context."
    )
    completion_tokens: int = Field(
        default=0, description="Tokens generated in response."
    )


class BaseLLMProvider(ABC):
    """Abstract Base Class for all LLM provider adapters."""

    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    async def generate(self, request: LLMRequest[T]) -> LLMResponse[T]:
        """Generates a completion from the LLM provider.

        If request.response_model is provided, the provider implementation must
        enforce JSON schema compliance and parse the output into that model.
        """
