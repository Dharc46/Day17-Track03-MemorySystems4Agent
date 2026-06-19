from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_dotenv(root / ".env")

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    default_models = {
        "openai": "gpt-4o-mini",
        "custom": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "anthropic": "claude-3-5-haiku-latest",
        "ollama": "llama3.1",
        "openrouter": "openai/gpt-4o-mini",
    }
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "custom": os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "ollama": None,
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
    }
    base_urls = {
        "custom": os.getenv("CUSTOM_BASE_URL"),
        "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "openrouter": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    }

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = ProviderConfig(
        provider=provider,
        model_name=os.getenv("LLM_MODEL", default_models[provider]),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        api_key=api_keys.get(provider),
        base_url=base_urls.get(provider),
    )
    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=os.getenv("JUDGE_MODEL", default_models[judge_provider]),
        temperature=float(os.getenv("JUDGE_TEMPERATURE", "0")),
        api_key=api_keys.get(judge_provider),
        base_url=base_urls.get(judge_provider),
    )

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "700")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "6")),
        model=model,
        judge_model=judge_model,
    )


def _load_dotenv(path: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path)
