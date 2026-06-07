"""Deterministic, evidence-backed memory extraction from scored sessions."""

from __future__ import annotations

from ..models import ConversationSession, EmotionalMoment, Memory, MemoryMoment, Message
from .semantics.evidence import rank_evidence_messages
from .semantics.phrases import extract_semantic_phrases


def _memory_type(session: ConversationSession, reconnecting: bool) -> str:
    signals = session.signal_counts
    if reconnecting:
        return "reconnection"
    if signals.get("fight") or signals.get("betrayal") or signals.get("distance"):
        return "conflict"
    if (
        signals.get("vulnerability")
        or signals.get("abandonment")
        or signals.get("insecurity")
        or signals.get("sadness")
    ) and (signals.get("support") or signals.get("reassurance")):
        return "comfort"
    if signals.get("vulnerability") or signals.get("abandonment") or signals.get("insecurity"):
        return "vulnerability"
    if signals.get("apology") and signals.get("reassurance"):
        return "reassurance"
    if signals.get("apology"):
        return "reassurance"
    if signals.get("support"):
        return "support"
    if signals.get("affection") or signals.get("openness"):
        return "confession"
    return "celebration"


def _summary(memory_type: str, evidence: list[Message], reconnecting: bool) -> str:
    first = evidence[0].text.strip()[:90] if evidence else ""
    second = evidence[1].text.strip()[:90] if len(evidence) > 1 else ""
    if reconnecting:
        return f"After a long pause, the conversation restarted around \"{first}\"."
    if second:
        return f"A {memory_type} moment: \"{first}\" was met with \"{second}\"."
    return f"A {memory_type} moment centered on \"{first}\"."


def extract_memories(
    sessions: list[ConversationSession],
    messages: list[Message],
) -> list[Memory]:
    memories: list[Memory] = []
    previous: ConversationSession | None = None
    for session in sessions:
        reconnecting = bool(previous and (session.start_time - previous.end_time).total_seconds() >= 48 * 3600)
        if session.score < 4 and not reconnecting:
            previous = session
            continue
        relevant_terms = {
            label.replace("_", " ")
            for label, count in session.signal_counts.items()
            if count and label not in {"late_night", "rapid_exchange", "filler", "logistics"}
        }
        ranked = rank_evidence_messages(session.messages, set(), [])
        evidence = [message for _, message in ranked[:3]]
        if not evidence:
            previous = session
            continue
        kind = _memory_type(session, reconnecting)
        model = EmotionalMoment if session.sentiment != 0 else MemoryMoment
        memories.append(model(
            summary=_summary(kind, evidence, reconnecting),
            emotional_weight=session.score + (2 if reconnecting else 0),
            evidence_messages=evidence,
            participants=session.participants,
            themes=sorted(relevant_terms)[:6],
            start_time=session.start_time,
            end_time=session.end_time,
            memory_type=kind,
            evidence_score=sum(score for score, _ in ranked[:3]),
        ))
        previous = session

    phrases = extract_semantic_phrases(messages, sessions)
    for phrase in phrases:
        if phrase["phrase_type"] != "relationship_specific" or phrase["score"] < 5:
            continue
        evidence = [
            message for message in messages
            if phrase["phrase"] in (message.normalized_text or message.text.casefold())
        ][:3]
        if evidence:
            memories.append(MemoryMoment(
                summary=f"They repeatedly returned to \"{phrase['phrase']}\", a shared reference specific to this chat.",
                emotional_weight=phrase["score"],
                evidence_messages=evidence,
                participants=phrase["senders"],
                themes=["inside joke", phrase["phrase"]],
                start_time=evidence[0].timestamp,
                end_time=evidence[-1].timestamp,
                memory_type="inside joke",
                evidence_score=phrase["score"],
            ))
    memories.sort(key=lambda memory: (-memory.emotional_weight, memory.start_time))
    return memories[:20]
