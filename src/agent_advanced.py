from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: advanced memory behavior.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Run the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)
        response_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens

        return {
            "agent": "advanced",
            "thread_id": thread_id,
            "answer": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "memory_path": str(self.profile_store.path_for(user_id)),
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """

        profile = self.profile_store.read_text(user_id)
        context = self.compact_memory.context(thread_id)
        recent = context.get("messages", [])
        recent_text = "\n".join(
            f"{m.get('role')}: {m.get('content')}" for m in recent if isinstance(m, dict)
        )
        full_context = "\n".join([profile, str(context.get("summary", "")), recent_text])
        return estimate_tokens(full_context)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """

        facts = self.profile_store.facts(user_id)
        lower = message.lower()

        if any(word in lower for word in ["tên", "nghề", "ở đâu", "nơi ở", "đồ uống", "món ăn", "style", "nuôi", "biết", "tóm tắt"]):
            parts = []
            if "tên" in lower or "biết" in lower or "tóm tắt" in lower:
                parts.append(_fmt("tên", facts.get("name")))
            if "nghề" in lower or "tóm tắt" in lower:
                parts.append(_fmt("nghề hiện tại", facts.get("profession")))
            if "ở đâu" in lower or "nơi ở" in lower or "hiện đang ở" in lower:
                parts.append(_fmt("nơi ở hiện tại", facts.get("location")))
            if "đồ uống" in lower:
                parts.append(_fmt("đồ uống yêu thích", facts.get("favorite_drink")))
            if "món ăn" in lower:
                parts.append(_fmt("món ăn yêu thích", facts.get("favorite_food")))
            if "style" in lower or "kiểu trả lời" in lower:
                parts.append(_fmt("style trả lời", facts.get("response_style")))
            if "nuôi" in lower or "con gì" in lower:
                parts.append(_fmt("thú nuôi", facts.get("pet")))
            if "quan tâm" in lower or "kỹ thuật" in lower or "tóm tắt" in lower:
                parts.append(_fmt("mối quan tâm kỹ thuật", facts.get("technical_interests")))

            usable = [part for part in parts if part]
            if usable:
                return "Mình nhớ: " + "; ".join(usable) + "."
            return "Mình chưa thấy fact ổn định nào trong User.md cho câu hỏi này."

        if extract_profile_updates(message):
            return "Mình đã cập nhật các thông tin ổn định vào User.md."

        if re.search(r"compact|token|benchmark|memory", lower):
            return "Gợi ý ngắn: advanced dùng User.md cho recall dài hạn và compact summary để giảm prompt tokens khi thread dài."

        return "Mình đã ghi nhận và sẽ ưu tiên các fact ổn định trong User.md."

    def _maybe_build_langchain_agent(self):
        """Build a live model hook for future agent/tool wiring.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None


def _fmt(label: str, value: str | None) -> str:
    if not value:
        return ""
    return f"{label}: {value}"
