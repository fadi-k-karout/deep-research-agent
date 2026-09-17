import os

from dotenv import load_dotenv


def load_environment() -> None:
    """Load variables from the project's .env file into the environment."""
    load_dotenv()


def _require(name: str, service: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} not set (required for {service})")
    return value


def tavily_api_key() -> str:
    return _require("TAVILY_API_KEY", "Tavily Search")


def exa_api_key() -> str:
    return _require("EXA_API_KEY", "Exa Search")
