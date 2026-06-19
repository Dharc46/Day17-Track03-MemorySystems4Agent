from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


def estimate_tokens(text: str) -> int:
    """Estimate tokens with a simple stable heuristic.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """

    cleaned = " ".join((text or "").split())
    if not cleaned:
        return 0
    return max(1, int(len(cleaned) / 4) + 1)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Responsibilities:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id.strip() or "anonymous")
        return self.root_dir / safe_id / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return _empty_profile(user_id)
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^-\s*([A-Za-z0-9_ -]+):\s*(.+)$", line.strip())
            if match:
                facts[match.group(1).strip().lower().replace(" ", "_")] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        content = self.read_text(user_id)
        key = key.strip().lower().replace(" ", "_")
        value = " ".join(value.split()).strip(" .")
        line = f"- {key}: {value}"
        pattern = re.compile(rf"^-\s*{re.escape(key)}\s*:\s*.+$", re.MULTILINE | re.IGNORECASE)
        if pattern.search(content):
            content = pattern.sub(line, content, count=1)
        else:
            if "## Stable facts" not in content:
                content = content.rstrip() + "\n\n## Stable facts\n"
            content = content.rstrip() + f"\n{line}\n"
        self.write_text(user_id, content)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    text = " ".join((message or "").split())
    lower = text.lower()
    if not text or (text.endswith("?") and not any(k in lower for k in ["mình tên", "mình ở", "mình là"])):
        return {}

    facts: dict[str, str] = {}

    patterns = [
        ("name", r"(?:mình\s+tên\s+là|tên\s+mình\s+là)\s+([^,.!?]+)"),
        ("location", r"(?:hiện\s+ở|đang\s+ở|mình\s+ở|nơi\s+ở\s+hiện\s+tại\s+là)\s+([^,.!?]+)"),
        ("profession", r"(?:đang\s+làm|mình\s+là|chuyển\s+sang)\s+([^,.!?]*(?:engineer|developer|manager|researcher|designer|student|sinh viên|MLOps engineer|backend engineer)[^,.!?]*)"),
        ("favorite_drink", r"(?:đồ\s+uống\s+yêu\s+thích\s+là|vẫn\s+uống)\s+([^,.!?]+)"),
        ("favorite_food", r"(?:món\s+ăn\s+yêu\s+thích\s+là|món\s+ruột)\s+([^,.!?]+)"),
        ("pet", r"(?:nuôi\s+(?:một\s+)?(?:bé\s+)?|con\s+)(corgi[^,.!?]*|Bơ[^,.!?]*)"),
    ]
    for key, pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            facts[key] = _clean_fact(match.group(1))

    if "ngắn gọn" in lower or "3 bullet" in lower or "bullet ngắn" in lower:
        style_bits = []
        if "3 bullet" in lower:
            style_bits.append("3 bullet")
        elif "bullet" in lower:
            style_bits.append("bullet ngắn")
        if "ngắn gọn" in lower:
            style_bits.append("ngắn gọn")
        if "ví dụ thực" in lower or "thực chiến" in lower or "thực tế" in lower:
            style_bits.append("có ví dụ thực tế")
        if "trade-off" in lower:
            style_bits.append("nhấn trade-off")
        facts["response_style"] = ", ".join(dict.fromkeys(style_bits))

    interests = [word for word in ["Python", "AI", "MLOps", "RAG", "evaluation"] if word.lower() in lower]
    if interests:
        facts["technical_interests"] = ", ".join(interests)

    if "không còn làm backend engineer" in lower or "đừng nói backend engineer" in lower:
        facts["profession"] = "MLOps engineer"
    if "huế sang đà nẵng" in lower or "nơi ở hiện tại là đà nẵng" in lower:
        facts["location"] = "Đà Nẵng"
    elif "giờ mình đang ở huế" in lower or "vẫn ở huế" in lower:
        facts["location"] = "Huế"

    return {key: value for key, value in facts.items() if value}


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    if not messages:
        return ""
    selected = messages[-max_items:]
    bullets = []
    for message in selected:
        role = message.get("role", "unknown")
        content = " ".join(message.get("content", "").split())
        if len(content) > 180:
            content = content[:177].rstrip() + "..."
        bullets.append(f"- {role}: {content}")
    return "Compact summary of older context:\n" + "\n".join(bullets)


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})

        total_text = str(thread.get("summary", "")) + "\n" + "\n".join(
            f"{m.get('role')}: {m.get('content')}" for m in messages
        )
        if estimate_tokens(total_text) <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        old_messages = messages[:-self.keep_messages]
        recent_messages = messages[-self.keep_messages :]
        previous_summary = str(thread.get("summary", "")).strip()
        new_summary = summarize_messages(old_messages)
        thread["summary"] = "\n".join(part for part in [previous_summary, new_summary] if part).strip()
        thread["messages"] = recent_messages
        thread["compactions"] = int(thread.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id).get("compactions", 0))


def _empty_profile(user_id: str) -> str:
    return f"# User profile: {user_id}\n\n## Stable facts\n"


def _clean_fact(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .")
    stop_words = [" chứ không", " nhưng", " và đang", " để "]
    lowered = value.lower()
    for marker in stop_words:
        idx = lowered.find(marker)
        if idx > 0:
            value = value[:idx].strip(" .")
            break
    return value
