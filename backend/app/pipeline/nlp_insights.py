"""
Shared NLP + evidence matching for Wrapped and Gift Engine.

No LLM — deterministic signals, human-readable narratives, quote grounding.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Any

import emoji as emoji_lib

from ..models import Message, MessageKind
from .content_filter import GENERIC_PHRASES, safe_for_display
from .stats import STOPWORDS

# Words that must never appear as "shared vocabulary" or persona anchors.
WEAK_TOKENS = set(
    """
    one two three four five six seven eight nine ten
    how what when where why who which that this those these
    now then just like get got going come went say said
    good great nice fine well really very also still even
    message edited deleted omitted media sticker image video
    call missed audio document forwarded
    """.split()
)

EXTRA_STOP = set(
    """
    with from into onto about after before because something anything
    everything today tomorrow yesterday actually literally maybe probably
    always never stuff things someone anyone everyone
    """.split()
)

ALL_STOP = STOPWORDS | EXTRA_STOP | WEAK_TOKENS

SYSTEM_NOISE_RE = re.compile(
    r"(?i)(this message was deleted|message deleted|you deleted this|"
    r"<media omitted>|image omitted|video omitted|missed voice call|"
    r"missed video call|sticker omitted|audio omitted|document omitted|"
    r"messages and calls are end-to-end|changed the subject|changed this group's icon)"
)

NIGHT_HOURS = {0, 1, 2, 3, 4, 5}
POSITIVE = {"love", "happy", "proud", "excited", "grateful", "thanks", "miss", "hug", "care", "proud"}
STRESS = {"stress", "stressed", "tired", "anxious", "sad", "angry", "deadline", "exam", "overwhelmed", "sorry", "cry"}
COMFORT = {"tea", "chai", "coffee", "sleep", "home", "calm", "okay", "fine", "warm", "cozy"}
SPIRITUAL = {"bible", "prayer", "church", "jesus", "god", "worship", "verse", "faith", "lord", "amen"}


def is_noise_message(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 2:
        return True
    if SYSTEM_NOISE_RE.search(text):
        return True
    if t in {"ok", "okay", "k", "kk", "hmm", "hm", "ya", "yes", "no"}:
        return True
    return False


def meaningful_tokens(text: str, *, min_len: int = 3) -> list[str]:
    text = emoji_lib.replace_emoji(text.lower(), replace=" ")
    out = []
    for word in re.findall(r"[a-z][a-z']{2,}", text):
        if word in ALL_STOP or len(word) < min_len:
            continue
        if word.isdigit():
            continue
        out.append(word)
    return out


def token_set(text: str) -> set[str]:
    return set(meaningful_tokens(text))


def evidence_score(quote: str, anchor_terms: set[str]) -> float:
    """Higher = quote actually supports the gift/insight."""
    if not quote or not anchor_terms:
        return 0.0
    if is_noise_message(quote):
        return -5.0
    q_tokens = token_set(quote)
    if not q_tokens:
        return 0.0
    overlap = q_tokens & anchor_terms
    if not overlap:
        return 0.0
    # Require at least one non-generic anchor hit
    strong = {t for t in overlap if t not in COMFORT | STRESS | POSITIVE}
    score = len(overlap) * 2.0 + len(strong) * 3.0
    # Bonus if anchor appears as substring (e.g. "bible" in "reading bible again")
    lower = quote.lower()
    for term in anchor_terms:
        if len(term) >= 4 and term in lower:
            score += 2.5
    # Penalize very long unrelated emotional dumps when anchor is narrow
    if len(q_tokens) > 25 and len(overlap) <= 1:
        score -= 2.0
    return score


def best_evidence_quote(
    examples: list[dict],
    anchor_terms: set[str],
    *,
    min_score: float = 3.0,
) -> tuple[str, str, float]:
    best_s, best_t, best_sc = "", "", -1.0
    for ex in examples:
        text = ex.get("text", "")
        matched = set(ex.get("matched", [])) | anchor_terms
        sc = evidence_score(text, matched)
        if sc > best_sc:
            best_sc, best_s, best_t = sc, ex.get("sender", ""), text[:200]
    if best_sc < min_score:
        return "", "", 0.0
    return best_s, best_t, best_sc


def extract_phrases(messages: list[Message], *, min_count: int = 2) -> list[dict[str, Any]]:
    """Rare bigrams/trigrams + emoji-adjacent phrases."""
    phrase_counts: Counter = Counter()
    phrase_quote: dict[str, str] = {}
    phrase_senders: dict[str, set] = defaultdict(set)

    for msg in messages:
        if msg.kind != MessageKind.TEXT or is_noise_message(msg.text):
            continue
        raw = msg.text.strip()
        # Emoji phrase: leading/trailing emoji with words
        emojis = _emojis(raw)
        tokens = meaningful_tokens(raw)
        if emojis and tokens:
            key = f"{' '.join(emojis[:2])} {' '.join(tokens[:3])}".strip()
            if len(key) > 4:
                phrase_counts[key] += 1
                phrase_quote.setdefault(key, raw[:160])
                phrase_senders[key].add(msg.sender)

        for n in (2, 3):
            for parts in zip(*(tokens[i:] for i in range(n))):
                phrase = " ".join(parts)
                if phrase in GENERIC_PHRASES or not safe_for_display(phrase):
                    continue
                if any(w in ALL_STOP for w in parts):
                    continue
                phrase_counts[phrase] += 1
                phrase_quote.setdefault(phrase, raw[:160])
                phrase_senders[phrase].add(msg.sender)

    scored = []
    for phrase, count in phrase_counts.items():
        if count < min_count:
            continue
        # Rarity: prefer longer phrases and multi-sender
        rarity = len(phrase.split()) + len(phrase_senders[phrase])
        scored.append((rarity * count, phrase, count, phrase_quote[phrase]))
    scored.sort(reverse=True)
    return [
        {"phrase": p, "count": c, "quote": q}
        for _, p, c, q in scored[:12]
    ]


def shared_vocabulary(messages: list[Message], senders: list[str]) -> list[dict[str, Any]]:
    per_sender: dict[str, Counter] = defaultdict(Counter)
    for msg in messages:
        if msg.kind == MessageKind.TEXT and not is_noise_message(msg.text):
            per_sender[msg.sender].update(meaningful_tokens(msg.text))

    if len(per_sender) < 2:
        return []

    shared = []
    all_words = set().union(*(c.keys() for c in per_sender.values()))
    for word in all_words:
        users = {s: per_sender[s][word] for s in per_sender if per_sender[s][word] > 0}
        if len(users) < 2:
            continue
        total = sum(users.values())
        if total < 4:
            continue
        shared.append({"word": word, "count": total, "senders": users})

    return sorted(shared, key=lambda x: (-x["count"], -len(x["word"])))[:30]


def nicknames(messages: list[Message], senders: list[str]) -> list[dict[str, Any]]:
    legal = {s.lower().split()[0] for s in senders if s}
    found: Counter = Counter()
    for msg in messages:
        if msg.kind != MessageKind.TEXT:
            continue
        for m in re.findall(r"\b([A-Z][a-z]{2,12})\b", msg.text):
            w = m.lower()
            if w not in legal and w not in ALL_STOP:
                found[w] += 1
        for m in re.findall(r"\b([a-z]{3,12}(?:u|ya|ie))\b", msg.text.lower()):
            if m not in legal and m not in ALL_STOP:
                found[m] += 1
    return [{"name": n, "count": c} for n, c in found.most_common(10) if c >= 2]


def _emojis(text: str) -> list[str]:
    return [e["emoji"] for e in emoji_lib.emoji_list(text)]


def persona_from_signals(
    messages: list[Message],
    phrases: list[dict],
    hour_counts: Counter,
) -> dict[str, Any]:
    text_msgs = [m for m in messages if m.kind == MessageKind.TEXT and not is_noise_message(m.text)]
    n = max(len(text_msgs), 1)
    night = sum(1 for m in text_msgs if m.timestamp.hour in NIGHT_HOURS) / n
    emoji_rate = sum(len(_emojis(m.text)) for m in text_msgs) / n
    vuln = sum(1 for m in text_msgs if token_set(m.text) & STRESS) / n
    support = sum(1 for m in text_msgs if token_set(m.text) & POSITIVE or "you got this" in m.text.lower()) / n
    humor = sum(1 for m in text_msgs if "haha" in m.text.lower() or "lol" in m.text.lower() or len(_emojis(m.text)) >= 2) / n
    deep = sum(1 for m in text_msgs if len(m.text) > 100 or token_set(m.text) & {"feel", "life", "trust", "future"}) / n
    spiritual = sum(1 for m in text_msgs if token_set(m.text) & SPIRITUAL) / n

    anchor_phrase = phrases[0]["phrase"] if phrases else ""
    # Never use noise as anchor
    if anchor_phrase and is_noise_message(anchor_phrase):
        anchor_phrase = phrases[1]["phrase"] if len(phrases) > 1 else ""

    candidates = [
        {
            "name": "The 2AM Safe Space",
            "score": night * 3 + vuln * 2 + support,
            "why": (
                "A real slice of this bond lives after midnight — replies get softer, "
                "and heavy messages rarely stay unanswered for long."
            ),
        },
        {
            "name": "The Chaos Duo",
            "score": emoji_rate * 2.5 + humor * 3,
            "why": (
                "Your chat runs on bursts, emojis, and quick callbacks — "
                "it feels loud and affectionate rather than formal."
            ),
        },
        {
            "name": "The Meme Ministers",
            "score": humor * 2 + (1 if anchor_phrase else 0) * 2,
            "why": (
                f"Recurring bits like “{anchor_phrase}” carry the relationship as much as serious talks."
                if anchor_phrase
                else "Running jokes and repeat phrases do more work here than long essays."
            ),
        },
        {
            "name": "The Midnight Philosophers",
            "score": deep * 3 + night * 1.5,
            "why": (
                "You drift into longer, reflective messages — especially when the day has already ended."
            ),
        },
        {
            "name": "The Soft Landing",
            "score": support * 3 + vuln,
            "why": (
                "When one of you wobbles, the other tends to answer with care quickly — "
                "reassurance is a reflex, not a performance."
            ),
        },
        {
            "name": "The Comfort Pair",
            "score": support * 2 + sum(1 for m in text_msgs if token_set(m.text) & COMFORT) / n * 2,
            "why": (
                "Small rituals — tea, check-ins, “you okay?” — show up often enough to be your real love language."
            ),
        },
    ]
    if spiritual > 0.08:
        candidates.append({
            "name": "The Sacred Thread",
            "score": spiritual * 5,
            "why": "Faith and spiritual check-ins weave through ordinary days — not just crisis moments.",
        })

    best = max(candidates, key=lambda c: c["score"])
    if anchor_phrase and best["name"] != "The Meme Ministers":
        best = {
            **best,
            "why": f"{best['why']} A phrase you keep returning to: “{anchor_phrase}”.",
        }
    return {"name": best["name"], "why": best["why"], "anchor_phrase": anchor_phrase}


def relationship_arc_events(messages: list[Message]) -> list[dict[str, Any]]:
    if len(messages) < 30:
        return [{
            "title": "Still writing your story",
            "why": "There is not enough history yet to mark clear chapters — but the rhythm is already forming.",
        }]

    # Monthly buckets with signals
    monthly: dict[str, list[Message]] = defaultdict(list)
    for m in messages:
        monthly[m.timestamp.strftime("%Y-%m")].append(m)

    months = sorted(monthly.keys())
    if len(months) <= 1:
        months = ["all"]
        monthly = {"all": messages}

    arc = []
    prev_avg_gap = None
    for i, month in enumerate(months):
        chunk = monthly[month]
        text = [m for m in chunk if m.kind == MessageKind.TEXT and not is_noise_message(m.text)]
        if not text:
            continue

        gaps = []
        for a, b in zip(chunk, chunk[1:]):
            gaps.append((b.timestamp - a.timestamp).total_seconds())
        silences = sum(1 for g in gaps if g > 3 * 86400)
        avg_gap = sum(gaps) / max(len(gaps), 1)
        stress_hits = sum(1 for m in text if token_set(m.text) & STRESS)
        support_hits = sum(1 for m in text if token_set(m.text) & POSITIVE)
        humor_hits = sum(1 for m in text if "haha" in m.text.lower() or "lol" in m.text.lower())
        night_hits = sum(1 for m in text if m.timestamp.hour in NIGHT_HOURS)

        title = f"Chapter {i + 1}"
        if i == 0:
            title = "When the thread started feeling real"
        elif i == len(months) - 1:
            title = "Where you are now"

        if silences >= 2 and (prev_avg_gap is None or avg_gap > prev_avg_gap * 1.3):
            narrative = (
                "There were noticeable silences here — then conversation came back denser, "
                "as if you were catching up on more than messages."
            )
        elif stress_hits > support_hits + 3:
            narrative = (
                "This stretch carries more pressure words than comfort — "
                "deadlines, worry, and shorter replies show up together."
            )
        elif support_hits > stress_hits + 5:
            narrative = (
                "This is where reassurance becomes dependable — "
                "check-ins land quickly and rarely feel performative."
            )
        elif humor_hits > len(text) * 0.15:
            narrative = (
                "Playful chaos dominates here — laughter and emoji carry the bond "
                "before anything gets too heavy."
            )
        elif night_hits > len(text) * 0.2:
            narrative = (
                "Late-night messages cluster in this phase — "
                "honesty shows up when the rest of the world is quiet."
            )
        else:
            narrative = (
                "The tone evens out into everyday rhythm — "
                "not a crisis arc, just the comfortable middle of knowing each other."
            )

        arc.append({
            "phase": month,
            "title": title,
            "messages": len(chunk),
            "why": narrative,
        })
        prev_avg_gap = avg_gap

    return arc[-6:]


def emotional_reading(messages: list[Message], top_supporter: str | None) -> dict[str, Any]:
    text = [m for m in messages if m.kind == MessageKind.TEXT and not is_noise_message(m.text)]
    stress_n = sum(1 for m in text if token_set(m.text) & STRESS)
    pos_n = sum(1 for m in text if token_set(m.text) & POSITIVE)
    comfort_n = sum(1 for m in text if token_set(m.text) & COMFORT)

    if pos_n >= stress_n + 10:
        headline = "Emotionally attentive"
        why = (
            "You both rarely leave stressful messages hanging. "
            "Reassurance shows up quickly — especially when someone sounds tired or unsure."
        )
    elif stress_n > pos_n + 10:
        headline = "Carrying a lot together"
        why = (
            "Pressure words outpace light banter in this export — "
            "the chat reads like mutual weather-reporting through hard weeks."
        )
    else:
        headline = "Balanced and lived-in"
        why = (
            "Neither all chaos nor all heaviness — "
            "ordinary life, jokes, and care share the same thread."
        )

    if top_supporter and comfort_n > 5:
        why += f" {top_supporter} often initiates the softer check-ins."

    return {
        "headline": headline,
        "why": why,
        "top_supporter": top_supporter,
        "comfort_language": _comfort_language(text),
    }


def _comfort_language(text_msgs: list[Message]) -> str:
    c = Counter()
    for m in text_msgs:
        c.update(token_set(m.text) & COMFORT)
    top = c.most_common(3)
    if not top:
        return "Small check-ins and quick replies"
    words = ", ".join(w for w, _ in top)
    return f"Tea, food, and calm words ({words}) appear when someone needs grounding"


def emotional_clock_narrative(messages: list[Message]) -> dict[str, Any]:
    buckets = {
        "vulnerable": (STRESS, "vulnerable"),
        "comfort": (COMFORT, "comforting"),
        "energy": ({"haha", "lol", "party", "yay"}, "high-energy"),
        "deep": ({"feel", "life", "trust", "future", "why"}, "reflective"),
    }
    hour_hits: dict[str, Counter] = {k: Counter() for k in buckets}

    for msg in messages:
        if msg.kind != MessageKind.TEXT or is_noise_message(msg.text):
            continue
        words = token_set(msg.text)
        h = msg.timestamp.hour
        for key, (lex, _) in buckets.items():
            if words & lex or (key == "energy" and len(_emojis(msg.text)) >= 2):
                hour_hits[key][h] += 1

    out = {}
    labels = {
        "vulnerable": "Emotionally raw hours",
        "comfort": "Comfort-window",
        "energy": "High-energy hours",
        "deep": "Deep-talk hours",
    }
    for key, counter in hour_hits.items():
        if not counter:
            out[key] = {"hour": None, "label": "—", "why": "No strong pattern in this export."}
            continue
        hour, count = counter.most_common(1)[0]
        _, mood = buckets[key]
        out[key] = {
            "hour": hour,
            "label": f"{hour:02d}:00",
            "why": (
                f"Around {hour:02d}:00, your chat tilts {mood} most often "
                f"({count} messages in this export carried that signal)."
            ),
        }
    return out


def viral_moments(
    messages: list[Message],
    phrases: list[dict],
    persona: dict,
    hour_counts: Counter,
) -> dict[str, Any]:
    text = [m for m in messages if m.kind == MessageKind.TEXT and not is_noise_message(m.text)]
    # Most human: moderate length, emotional tokens, not noise
    scored = []
    for m in text:
        w = token_set(m.text)
        if not w:
            continue
        score = len(w & (POSITIVE | STRESS | COMFORT)) * 2 + min(len(m.text), 200) / 50
        if 40 < len(m.text) < 220:
            score += 2
        scored.append((score, m))
    scored.sort(key=lambda x: -x[0])
    human = scored[0][1] if scored else None

    peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else 12
    vuln_phrases = [p for p in phrases if p["count"] >= 2][:1]

    movie = persona.get("name", "Your bond")
    if phrases:
        movie = f"If your chat was a movie: {movie} — with running gags about “{phrases[0]['phrase']}”."

    return {
        "most_human_moment": {
            "quote": human.text[:220] if human else "",
            "sender": human.sender if human else "",
            "why": "The message that sounds most like you two at your realest — not the longest, the most emotionally specific.",
        },
        "emotional_timezone": {
            "hour": peak_hour,
            "label": f"{peak_hour:02d}:00",
            "why": f"Most messages land around {peak_hour:02d}:00 — your shared emotional timezone in this export.",
        },
        "comfort_language": emotional_reading(messages, None).get("comfort_language", ""),
        "movie_tagline": movie,
        "vulnerable_phrase": vuln_phrases[0] if vuln_phrases else None,
    }


def dedupe_gifts(ideas: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out = []
    for idea in ideas:
        title_key = re.sub(r"\W+", "", idea.get("title", "").lower())[:50]
        quote_key = re.sub(r"\W+", "", idea.get("quote", "").lower())[:40]
        composite = f"{title_key}:{quote_key}"
        if composite in seen or title_key in seen:
            continue
        seen.add(composite)
        seen.add(title_key)
        out.append(idea)
    return out
