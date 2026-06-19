from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider aliases like `anthorpic` to `anthropic`."""

    normalized = (value or "openai").strip().lower().replace("-", "_")
    aliases = {
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "google": "gemini",
        "google_genai": "gemini",
        "open_ai": "openai",
        "openrouter": "openrouter",
        "open_router": "openrouter",
        "local": "ollama",
        "openai_compatible": "custom",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if normalized not in supported:
        raise ValueError(f"Unsupported provider {value!r}. Supported: {sorted(supported)}")
    return normalized


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """

    provider = normalize_provider(config.provider)
    kwargs = {"model": config.model_name, "temperature": config.temperature}

    if provider in {"openai", "custom"}:
        from langchain_openai import ChatOpenAI

        if config.api_key:
            kwargs["api_key"] = config.api_key
        if provider == "custom" and config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError:
            from langchain_openai import ChatOpenAI

            kwargs["base_url"] = config.base_url or "https://openrouter.ai/api/v1"
            if config.api_key:
                kwargs["api_key"] = config.api_key
            return ChatOpenAI(**kwargs)

        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenRouter(**kwargs)

    raise ValueError(f"Unsupported provider {provider!r}")
