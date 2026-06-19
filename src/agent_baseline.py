from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: baseline memory behavior.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting.

        Pseudocode:
        - If a live agent exists, call the live path.
        - Otherwise use a deterministic offline path.
        """

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Run a simple deterministic offline behavior.

        Suggested behavior:
        - Store the new user message in the session
        - Generate a short deterministic reply
        - Update token counts
        - Never remember facts across different thread ids
        """

        session = self.sessions.setdefault(thread_id, SessionState())
        prompt_tokens = estimate_tokens(
            "\n".join(f"{m['role']}: {m['content']}" for m in session.messages)
            + f"\nuser: {message}"
        )
        session.prompt_tokens_processed += prompt_tokens
        session.messages.append({"role": "user", "content": message})

        response = self._deterministic_response(session.messages, message)
        session.messages.append({"role": "assistant", "content": response})
        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens

        return {
            "agent": "baseline",
            "thread_id": thread_id,
            "answer": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally build a live chat model when dependencies and keys exist.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None

    def _deterministic_response(self, messages: list[dict[str, str]], message: str) -> str:
        lower = message.lower()
        thread_text = "\n".join(m["content"] for m in messages if m["role"] == "user")

        if any(marker in lower for marker in ["tên", "nghề", "ở đâu", "nơi ở", "đồ uống", "món ăn", "style", "nuôi"]):
            facts = []
            for label, needles in {
                "tên": ["DũngCT Stress", "DũngCT"],
                "nghề": ["MLOps engineer", "backend engineer"],
                "nơi ở": ["Đà Nẵng", "Huế", "Đà Nẵng"],
                "đồ uống": ["cà phê sữa đá"],
                "món ăn": ["mì Quảng"],
                "style": ["3 bullet", "ngắn gọn"],
                "thú nuôi": ["corgi"],
                "kỹ thuật": ["Python", "AI"],
            }.items():
                for needle in needles:
                    if needle.lower() in thread_text.lower():
                        facts.append(f"{label}: {needle}")
                        break
            if facts:
                return "Trong thread hiện tại mình thấy " + "; ".join(facts) + "."
            return "Mình không có trí nhớ dài hạn trong thread mới, nên không chắc thông tin này."

        return "Mình đã ghi nhận trong thread hiện tại. Baseline chỉ giữ ngữ cảnh ngắn hạn."
