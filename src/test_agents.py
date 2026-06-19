from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    """Build an isolated config for tests."""

    config = load_config(Path(__file__).resolve().parent.parent)
    config.state_dir = tmp_path / "state"
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.compact_threshold_tokens = 120
    config.compact_keep_messages = 4
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")
    assert "Stable facts" in store.read_text("student")
    path = store.write_text("student", "# User profile: student\n\n## Stable facts\n- name: An\n")
    assert path.exists()
    assert store.edit_text("student", "An", "Binh") is True
    assert "Binh" in store.read_text("student")
    assert store.file_size("student") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for index in range(12):
        agent.reply("u1", "long-thread", f"Turn {index}: " + "context " * 80)
    context = agent.compact_memory.context("long-thread")
    assert agent.compaction_count("long-thread") > 0
    assert context["summary"]
    assert len(context["messages"]) <= agent.config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    advanced.reply("u2", "session-a", "Mình tên là DũngCT và mình đang làm MLOps engineer.")
    baseline.reply("u2", "session-a", "Mình tên là DũngCT và mình đang làm MLOps engineer.")

    advanced_answer = advanced.reply("u2", "session-b", "Mình tên gì và hiện tại mình làm nghề gì?")["answer"]
    baseline_answer = baseline.reply("u2", "session-b", "Mình tên gì và hiện tại mình làm nghề gì?")["answer"]

    assert "DũngCT" in advanced_answer
    assert "MLOps engineer" in advanced_answer
    assert "DũngCT" not in baseline_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    for index in range(18):
        message = f"Long turn {index}: " + "memory benchmark context " * 70
        baseline.reply("u3", "thread", message)
        advanced.reply("u3", "thread", message)

    assert advanced.compaction_count("thread") > 0
    assert advanced.prompt_token_usage("thread") < baseline.prompt_token_usage("thread")
