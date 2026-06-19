from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    """Return a proportional recall score based on expected facts."""

    if not expected:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for item in expected if item.lower() in answer_lower)
    return hits / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Compute a lightweight quality score for offline mode."""

    if not answer.strip():
        return 0.0
    recall = recall_points(answer, expected)
    concise_bonus = 0.2 if len(answer) <= 400 else 0.0
    structure_bonus = 0.1 if any(mark in answer for mark in [";", "-", ":"]) else 0.0
    return min(1.0, 0.6 * recall + concise_bonus + structure_bonus)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    agent_tokens = 0
    prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    compactions = 0
    memory_before = 0
    memory_after = 0

    users = {conversation["user_id"] for conversation in conversations}
    if hasattr(agent, "memory_file_size"):
        memory_before = sum(agent.memory_file_size(user_id) for user_id in users)

    for conversation in conversations:
        user_id = conversation["user_id"]
        thread_id = conversation["id"]
        for turn in conversation.get("turns", []):
            result = agent.reply(user_id, thread_id, turn)
            agent_tokens += int(result.get("agent_tokens", 0))
            prompt_tokens += int(result.get("prompt_tokens", 0))

        compactions += agent.compaction_count(thread_id)
        for index, question in enumerate(conversation.get("recall_questions", [])):
            recall_thread = f"{thread_id}-recall-{index}"
            result = agent.reply(user_id, recall_thread, question["question"])
            answer = result.get("answer", "")
            expected = question.get("expected_contains", [])
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))
            agent_tokens += int(result.get("agent_tokens", 0))
            prompt_tokens += int(result.get("prompt_tokens", 0))

    if hasattr(agent, "memory_file_size"):
        memory_after = sum(agent.memory_file_size(user_id) for user_id in users)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=max(0, memory_after - memory_before),
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a markdown table."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row.agent_name,
            str(row.agent_tokens_only),
            str(row.prompt_tokens_processed),
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            str(row.memory_growth_bytes),
            str(row.compactions),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    """Run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)

    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")

    with tempfile.TemporaryDirectory(prefix="memory_lab_benchmark_") as tmp:
        tmp_root = Path(tmp)

        print("## Standard Benchmark")
        standard_rows = [
            run_agent_benchmark(
                "Baseline",
                BaselineAgent(_isolated_config(config, tmp_root, "standard_baseline"), force_offline=True),
                standard,
                config,
            ),
            run_agent_benchmark(
                "Advanced",
                AdvancedAgent(_isolated_config(config, tmp_root, "standard_advanced"), force_offline=True),
                standard,
                config,
            ),
        ]
        print(format_rows(standard_rows))

        print("\n## Long-Context Stress Benchmark")
        stress_rows = [
            run_agent_benchmark(
                "Baseline",
                BaselineAgent(_isolated_config(config, tmp_root, "stress_baseline"), force_offline=True),
                stress,
                config,
            ),
            run_agent_benchmark(
                "Advanced",
                AdvancedAgent(_isolated_config(config, tmp_root, "stress_advanced"), force_offline=True),
                stress,
                config,
            ),
        ]
        print(format_rows(stress_rows))


def _isolated_config(config, tmp_root: Path, name: str):
    state_dir = tmp_root / name
    state_dir.mkdir(parents=True, exist_ok=True)
    return replace(config, state_dir=state_dir)


if __name__ == "__main__":
    main()
