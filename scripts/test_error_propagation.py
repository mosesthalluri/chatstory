"""Test that LLM connection failures NEVER appear as content.

The user reported: "the output summary made a story that ollama connection
failed?, how can llm make a story if connection failed?"

This test verifies the fix: every place that calls the LLM now either:
  - Returns None on single-call failure (per-call helpers)
  - Raises LLMError on persistent failure (so orchestrator catches it)
  - Returns a neutral fallback (NOT containing the error message)

Run: python scripts/test_error_propagation.py
"""
import sys, types, asyncio
from pathlib import Path
from datetime import date, datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.modules["httpx"] = types.ModuleType("httpx")
pyds = types.ModuleType("pydantic_settings")
pyds.BaseSettings = type("BaseSettings", (), {})
pyds.SettingsConfigDict = lambda **kw: {}
sys.modules["pydantic_settings"] = pyds

from app.models import Message, MessageKind
from app import llm as llm_mod
from app.pipeline import summarize, chapter_gen


# ---------------------------------------------------------------------------
# Simulate Ollama being unreachable
# ---------------------------------------------------------------------------
async def always_fail(*args, **kwargs):
    raise llm_mod.LLMError(
        "Could not reach Ollama at http://localhost:11434. "
        "Is `ollama serve` running?"
    )


# Patch the LLM client so every call fails like Ollama being down
llm_mod.complete = always_fail


async def main():
    # Test 1: day summary returns None on failure (not error string)
    print("=== Test 1: summarize_day returns None on failure ===")
    msgs = [
        Message(
            sender="alice", timestamp=datetime(2025, 8, 8, 10, 0),
            text="hello there", kind=MessageKind.TEXT,
        )
        for _ in range(10)
    ]
    result = await summarize.summarize_day(date(2025, 8, 8), msgs)
    print(f"Result: {result!r}")
    assert result is None, (
        f"summarize_day must return None on LLM failure, not '{result}'"
    )
    assert result is None or "(" not in (result or ""), \
        "summarize_day must not return error strings"
    print("✓ Day summary returns None cleanly (no error string in return)\n")

    # Test 2: rollup returns None on failure
    print("=== Test 2: rollup returns None on failure ===")
    result = await summarize.rollup("week", "some summaries text")
    print(f"Result: {result!r}")
    assert result is None
    print("✓ Rollup returns None cleanly\n")

    # Test 3: arc returns a neutral fallback, NOT the error message
    print("=== Test 3: identify_arc returns neutral fallback ===")
    result = await summarize.identify_arc(
        {date(2025, 8, 1): "August was a busy month."}, span_days=30
    )
    print(f"Result: {result!r}")
    assert result, "Arc should return SOMETHING for chapter generator"
    assert "Ollama" not in result, (
        f"Arc must not leak error message into content: {result}"
    )
    assert "Could not reach" not in result
    assert "failed" not in result.lower() or "unavailable" in result.lower()
    print("✓ Arc returns neutral fallback (no error message leaked)\n")

    # Test 4: chapter generation RAISES instead of returning fake chapter
    print("=== Test 4: chapter_gen.generate_chapter raises LLMError ===")
    try:
        chap = await chapter_gen.generate_chapter(
            index=1,
            start_date=date(2025, 8, 8),
            end_date=date(2025, 8, 8),
            chapter_messages=msgs,
            month_summaries={},
            arc_context="some arc",
        )
        # If we got here, the LLM error was swallowed — BUG!
        print(f"BUG: got a Chapter back instead of an error: {chap}")
        print(f"Body: {chap.body[:100]!r}")
        assert False, "Chapter generation must raise on LLM failure, not return fake Chapter"
    except llm_mod.LLMError as e:
        print(f"Caught: {e}")
        print("✓ Chapter generation propagates LLMError instead of returning a chapter with error in body\n")

    # Test 5: summarize_all_days raises after too many failures
    print("=== Test 5: summarize_all_days raises if all days fail ===")
    msgs_by_day = []
    for day in range(1, 11):
        for _ in range(8):
            msgs_by_day.append(Message(
                sender="alice",
                timestamp=datetime(2025, 8, day, 10, 0),
                text="hello", kind=MessageKind.TEXT,
            ))
    try:
        await summarize.summarize_all_days(msgs_by_day, max_concurrent=2)
        assert False, "Should have raised LLMError after all days failed"
    except llm_mod.LLMError as e:
        print(f"Caught: {e}")
        assert "unreachable" in str(e).lower() or "failed" in str(e).lower()
        print("✓ summarize_all_days raises clean error when LLM is dead\n")

    print("✓ ERROR PROPAGATION TEST PASSED")
    print()
    print("Conclusion: When the LLM connection fails, the user will now see")
    print("a clear error state instead of a 'book' describing the failure.")


asyncio.run(main())
