"""
LLM gift personalization (optional, Ollama/Groq via app.llm).

The deterministic engine matches keywords; this layer adds *intelligence*:
it reads real chat lines, infers what each person is actually like (their
hobbies, skills, identity, love language), and proposes specific, personal
gift ideas — a mix of things to buy, handmade/DIY, gestures (write her a
song if he sings, make art if she paints), and experiences.

Hallucination guard: the model is given REAL numbered quotes and must cite
the quote index that justifies each idea. We attach the real quote by that
index and drop any idea that cites nothing valid — so quotes are never made
up, even though the ideas are creative.
"""

from __future__ import annotations

import json
import re

from .. import llm

PROMPT = """You are a thoughtful, creative gift advisor. Two people chat; below are
REAL lines from their conversation, each with an index in [brackets].

People: {names}

REAL CHAT LINES:
{quotes}

From ONLY what these lines reveal, infer what each person is actually like —
their hobbies, skills, talents, tastes, worries, and how they show love.

Then propose {n} gift ideas that feel personal and specific to THESE people —
not generic. Mix the kinds of gift:
  - "buy": a specific product to purchase
  - "make": a handmade / DIY gift
  - "gesture": an act or creation that fits a talent (e.g. write her a song if
    he sings, paint something if she draws, cook a dish they love)
  - "experience": something to do together

Rules:
- Every idea MUST cite the quote index/indices that justify it (the evidence).
- Be concrete and unique. Avoid generic "a mug" / "a journal" unless the chat
  truly points to it.
- Say who it's FOR and a warm one-line reason tied to the evidence.

Respond as STRICT JSON only:
{{"ideas": [
  {{"title": "...", "type": "buy|make|gesture|experience", "for": "<name>",
    "reason": "...", "evidence": [<index>, ...]}}
]}}
"""


def _parse_json(text: str) -> dict:
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


async def personalize_gifts(senders: list[str], evidence_pool: list[dict],
                            n: int = 6) -> list[dict]:
    """Return a list of grounded, personalized gift dicts (or [] on failure)."""
    if not evidence_pool:
        return []

    numbered = "\n".join(f"[{i}] {e['sender']}: {e['text']}"
                         for i, e in enumerate(evidence_pool))
    prompt = PROMPT.format(names=" & ".join(senders[:2]) or "two friends",
                           quotes=numbered, n=n)

    response = await llm.complete(
        [
            {"role": "system", "content": "You are a creative, grounded gift advisor. "
             "You only use the evidence provided and always cite it. Output strict JSON."},
            {"role": "user", "content": prompt},
        ],
        model_size="strong", temperature=0.6, max_tokens=900,
    )

    data = _parse_json(response)
    ideas_in = data.get("ideas") if isinstance(data, dict) else None
    if not isinstance(ideas_in, list):
        return []

    out: list[dict] = []
    for raw in ideas_in:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title", "")).strip()
        reason = str(raw.get("reason", "")).strip()
        if not title or not reason:
            continue
        # Ground the citation: keep only valid indices, attach the real quote.
        idxs = raw.get("evidence", [])
        if isinstance(idxs, int):
            idxs = [idxs]
        valid = [i for i in idxs if isinstance(i, int) and 0 <= i < len(evidence_pool)]
        if not valid:
            continue  # no real evidence cited -> drop (anti-hallucination)
        ev = evidence_pool[valid[0]]
        gtype = str(raw.get("type", "")).lower()
        if gtype not in {"buy", "make", "gesture", "experience"}:
            gtype = "gesture"
        out.append({
            "title": title[:160],
            "type": gtype,
            "for": str(raw.get("for", "")).strip()[:60],
            "reason": reason[:300],
            "quote": ev["text"],
            "quote_sender": ev["sender"],
            "confidence": min(95, 70 + len(valid) * 6),
        })
        if len(out) >= n:
            break
    return out
