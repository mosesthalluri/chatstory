"""Focused deterministic checks for the reusable relationship-intelligence pipeline."""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core import build_intelligence
from app.services.chat_wrapped import compute_wrapped
from app.services.gift_engine import compute_gifts
from app.models import Message


def _message(hour: int, minute: int, sender: str, text: str, day: int = 1) -> Message:
    return Message(
        sender=sender,
        timestamp=datetime(2025, 1, day, hour, minute),
        text=text,
    )


def main() -> None:
    messages = [
        _message(22, 10, "A", "Nuvvu ante istam!!! I miss u"),
        _message(22, 11, "B", "I love you too, main hu na"),
        _message(22, 12, "A", "Thanku, I was scared"),
        _message(9, 0, "B", "gm", day=2),
        _message(9, 1, "A", "okay", day=2),
        _message(20, 0, "A", "I am sorry, I hurt you", day=3),
        _message(20, 2, "B", "We can talk, do not worry", day=3),
    ]

    result = build_intelligence(messages, ["A", "B"])

    assert messages[0].normalized_text
    assert "love you" in messages[0].normalized_text
    assert len(result.sessions) == 3
    assert len(result.selected_sessions) == 2
    assert result.selected_sessions[0].score > result.sessions[1].score
    assert result.analytics["relationship_timeline"]
    assert result.analytics["emotional_trend"]["peak_session"]
    assert result.analytics["silence_drift"]["longest_silences"]
    assert result.analytics["communication_rhythm"]["initiations"]
    assert result.memories

    meaning_messages = [
        _message(10, 0, "A", "https://www.instagram.com/reel/abc forwarded attachment"),
        _message(10, 1, "B", "https://www.instagram.com/reel/abc forwarded attachment"),
        _message(22, 0, "A", "Proud of u, mooncake promise forever", day=2),
        _message(22, 1, "B", "Proud of u, mooncake promise forever", day=2),
        _message(20, 0, "A", "Don't leave, mooncake promise forever", day=5),
        _message(20, 1, "B", "I am here for you, mooncake promise forever", day=5),
    ]
    meaningful = build_intelligence(meaning_messages, ["A", "B"])
    phrases = meaningful.semantic_phrases
    phrase_text = " ".join(phrase["phrase"] for phrase in phrases)
    assert "https" not in phrase_text
    assert "instagram" not in phrase_text
    assert any(phrase["phrase"] == "proud of you" for phrase in phrases)
    assert any(phrase["phrase"] == "mooncake" for phrase in phrases)
    assert any(memory.memory_type == "reconnection" for memory in meaningful.memories)
    inside_memories = [memory.summary for memory in meaningful.memories if memory.memory_type == "inside joke"]
    assert all("proud of" not in summary and "of you" not in summary for summary in inside_memories)

    wrapped = compute_wrapped(meaning_messages, "test", ["A", "B"])
    wrapped_terms = " ".join(item["word"] for item in wrapped["shared_vocabulary"])
    assert "instagram" not in wrapped_terms
    assert any(item["phrase"] == "mooncake" for item in wrapped["inside_jokes"])
    assert wrapped["strongest_moments"]

    gifts = compute_gifts(meaning_messages, "test", ["A", "B"])
    suggestions = [
        gift for group in gifts["suggestions"].values() for gift in group
    ]
    assert all("instagram" not in gift.get("title", "").casefold() for gift in suggestions)
    assert any("mooncake" in gift.get("title", "").casefold() for gift in gifts["suggestions"]["inside_jokes"])
    assert gifts["relationship_intelligence"]["memories"]

    print("core intelligence test passed")


if __name__ == "__main__":
    main()
