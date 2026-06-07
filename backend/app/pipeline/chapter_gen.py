"""
Chapter generation. Builds a tightly-grounded prompt and writes one
chapter per call.

This module is the highest-stakes part of the pipeline because LLMs
LOVE to invent narrative detail. Our defenses, in order of importance:

  1. Tagged source format. MEDIA_PLACEHOLDER messages are passed as
     [SHARED MEDIA: caption] so the LLM knows the text is a reel/song
     caption, not something the sender wrote.
  2. Entity scaffolding. We deterministically extract nicknames, real
     names, and third-party references from messages and pass them as
     a "people mentioned" list. The LLM should use these specifics
     instead of inventing generic ones.
  3. Explicit date boundaries. The prompt names the exact date range
     and forbids mentioning any other date.
  4. Adaptive length. With few messages, we ask for a shorter chapter.
     A 350-word chapter from 15 messages forces invention.
  5. No-interpretation rule. The prompt forbids inferring emotional
     states or motivations not directly stated in messages.
  6. Pull-quote validation. The chosen quote must be a real text
     message (not a media share), pass the content filter, and verify
     against the actual message list.
"""

import json
import re
from datetime import date
from typing import NamedTuple

from .. import llm
from ..models import Message, MessageKind
from . import highlights


class Chapter(NamedTuple):
    index: int
    title: str
    when: str
    body: str
    pull_quote: str
    pull_quote_author: str
    illustration_prompt: str


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------
# We pre-extract specific references the LLM should use rather than
# inventing generic narrative. This is the difference between
# "trust issues" (invented psychology) and "her sister was in the room"
# (a real reason from the messages).

# Hindi/Urdu kinship terms. Common in Indian chats. Each indicates a real
# third party present in the relationship's life that the book should
# incorporate. Add more as we encounter real edge cases.
HINDI_RELATIONS = {
    "didi": "elder sister",
    "behen": "sister", "bhen": "sister",
    "bhaiya": "elder brother", "bhai": "brother",
    "mummy": "mother", "ma": "mother", "maa": "mother",
    "papa": "father", "pita": "father",
    "nani": "maternal grandmother", "dadi": "paternal grandmother",
    "nana": "maternal grandfather", "dada": "paternal grandfather",
    "mami": "maternal uncle's wife", "mausi": "maternal aunt",
    "bua": "paternal aunt", "chacha": "paternal uncle",
    "chachi": "paternal uncle's wife", "bhabhi": "sister-in-law",
    "jiju": "brother-in-law", "saala": "wife's brother",
}

# English relations
ENGLISH_RELATIONS = {
    "mom": "mother", "mum": "mother", "mommy": "mother",
    "dad": "father", "daddy": "father",
    "sister": "sister", "sis": "sister",
    "brother": "brother", "bro": "brother",
    "aunt": "aunt", "uncle": "uncle",
    "grandma": "grandmother", "grandpa": "grandfather",
    "cousin": "cousin",
}


def _extract_entities(
    messages: list[Message],
    sender_names: set[str],
) -> dict[str, list[str]]:
    """Find nicknames, real names, and third-party references in the
    messages. These get passed to the chapter prompt so the LLM uses
    real specifics from the source instead of inventing them.

    Returns a dict with:
        nicknames_for_each_other: list of vocative addresses each used
        other_people_mentioned: list of "name (relation)" strings
        proper_nouns: capitalized words that aren't the senders'
    """
    nicknames: set[str] = set()
    other_people: set[str] = set()  # "didi (elder sister)", "Himanshu", etc.
    proper_nouns: set[str] = set()

    # Lowercase sender names for comparison
    sender_lower = {s.lower() for s in sender_names}
    # Also strip emojis from sender names so "✨Kristy_honey✨" matches "kristy_honey"
    sender_lower_clean = {
        re.sub(r"[^\w]", "", s).lower() for s in sender_names
    }

    for m in messages:
        if m.kind != MessageKind.TEXT:
            continue
        text = m.text

        # Hindi/English relation terms — directly indicate third parties
        words_lower = re.findall(r"[a-zA-Z]+", text.lower())
        for w in words_lower:
            if w in HINDI_RELATIONS:
                other_people.add(f"{w} ({HINDI_RELATIONS[w]})")
            elif w in ENGLISH_RELATIONS and len(w) > 2:
                # avoid catching "mum"/"dad" in mid-word
                if re.search(rf"\b{w}\b", text.lower()):
                    other_people.add(f"{w} ({ENGLISH_RELATIONS[w]})")

        # Proper nouns (capitalized words not at sentence start, length 3+)
        # Skip the senders' own names.
        for match in re.finditer(r"\b[A-Z][a-zA-Z]{2,}\b", text):
            word = match.group()
            word_lower = word.lower()
            if word_lower in sender_lower:
                continue
            if re.sub(r"[^\w]", "", word_lower) in sender_lower_clean:
                continue
            # Skip common English words that happen to be capitalized
            if word_lower in {"the", "and", "but", "yes", "no", "ok",
                              "okay", "good", "morning", "night", "love",
                              "hey", "hi", "hello", "please", "sorry"}:
                continue
            # Skip common Hindi/Urdu sentence-start words (false positives)
            if word_lower in {
                "aur", "ya", "haan", "haa", "han", "nahi", "nhi", "kuch",
                "kya", "kab", "kahan", "kaun", "kaisa", "kaise",
                "mai", "main", "mein", "tum", "tu", "aap", "hum", "yeh",
                "yh", "woh", "wo", "vo", "mere", "mera", "meri", "tera",
                "teri", "tumhara", "tumhari", "iska", "uska", "sab",
                "bhi", "phir", "fir", "abhi", "ab", "toh", "to", "agar",
                "magar", "lekin", "par", "ki", "ke", "ka", "se", "ne",
                "hota", "hoti", "hua", "hui", "tha", "thi", "honge",
                "achha", "acha", "thik", "theek", "baby", "babu", "jaan",
                "ji", "haaa", "huh", "uhh", "oye", "arey", "are", "abe",
                # Verb-forms and imperatives that get capitalized at sentence start
                "bol", "batao", "bhejo", "bhejna", "bhej", "dekho", "dekha",
                "suno", "sun", "ja", "jaa", "jao", "aaja", "aja",
                "kr", "kar", "karo", "karna", "kre", "krte", "krna",
                "ho", "hoo", "hua", "hui", "hain", "hu", "raha", "rahi",
                "le", "lo", "li", "leke",
                "ruk", "ruko", "rukna", "rok", "roko",
                "soo", "so", "sona", "soya", "soye",
                # Common Hindi adjectives/adverbs that get capitalized
                "adat", "aisa", "aise", "itna", "utna", "yahi", "wahi",
                "bhut", "bohot", "thoda", "zyada", "kam", "bas", "bs",
                "khair", "okay", "ok", "yes", "yeah", "uh", "umm",
                "call", "phone", "text", "msg", "pic", "photo",
                "baat", "batein", "dil", "dilse", "nind", "khwab",
                "joo", "jo", "voh",
            }:
                continue
            proper_nouns.add(word)

        # Vocative nickname detection: if a message starts with a name-like
        # word followed by a comma or other terms ("Mr rabbit", "Mr patel",
        # "babu please"), treat as a possible nickname.
        # Pattern: "Mr X" or "babu" or a name from proper_nouns used in
        # second-person context.
        mr_match = re.search(r"\bMr\s+([A-Za-z]+)", text)
        if mr_match:
            nick = f"Mr {mr_match.group(1).lower().capitalize()}"
            nicknames.add(nick)
        # Common Indian terms of endearment
        for endearment in ["babu", "baby", "jaan", "shona", "raja", "rani"]:
            if re.search(rf"\b{endearment}\b", text.lower()):
                nicknames.add(endearment)

    return {
        "nicknames": sorted(nicknames),
        "other_people": sorted(other_people),
        "proper_nouns": sorted(proper_nouns)[:10],  # cap at top 10
    }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

CHAPTER_PROMPT = """You are writing one chapter of a book about a personal
chat relationship. The book is honest, warm, third-person, biographer-style.

== CHAPTER BOUNDARIES (DO NOT EXCEED) ==
This chapter covers ONLY messages from: {date_range_text}
Do not refer to any other dates. Do not invent dates that span beyond this.
Do not mention "weeks of" or "months of" unless the range above explicitly
includes that span.

== BOOK ARC CONTEXT ==
{arc_context}

== PERIOD CONTEXT ==
{month_summaries}

== PEOPLE IN THIS CHAPTER ==
The two people whose chat this is:
{sender_list}

{entities_block}

== ACTUAL MESSAGES (your ONLY source of truth) ==
Below is the conversation between these two people. Most of these are
short back-and-forth chat messages — that IS the conversation, and that
is what the chapter should narrate. Even one-word replies ("Hmm", "Hnn",
"Mr Rabbit", "Nahi") are part of the dialogue and contribute to the
emotional arc.

Lines prefixed with [SHARED MEDIA] are reels, songs, or videos that
person SHARED — they did NOT type those words. They are CONTEXT, not
the heart of the conversation. You may mention them briefly if relevant,
but the chapter is about the actual back-and-forth, not the shared
content.

{highlight_block}

== HARD RULES ==
1. NARRATE THE CONVERSATION. The chapter must describe the back-and-forth
   between the two people — what they argued about, what one said and how
   the other replied, what specific concerns or requests came up. If the
   messages show a fight about a photo with a sibling in the room, the
   chapter should mention that fight, the photo, and the sibling. Do not
   write a chapter that only mentions media-shares while ignoring 100
   text messages of dialogue.
2. ONLY narrate what the messages explicitly show. Do not infer emotions
   or motivations beyond what someone literally states. If no message
   mentions "trust", do not write about "trust issues." If no message
   mentions "trauma", do not invent past trauma.
3. Use the people's actual names and the nicknames listed above. If the
   "Other people mentioned" list includes family or friends, weave them
   in where the messages support it.
4. Length should match the material. If there are 100+ text messages,
   write a substantial chapter. If there are 10 messages, write a short
   one. Do not pad. Do not invent to fill space.
5. Do not write generic relationship prose ("complex emotions",
   "unspoken tension", "intricate web of feelings"). Be specific or be
   silent.

== OUTPUT FORMAT ==
Respond with EXACTLY these section headers, in this order. Use plain
text — do NOT add markdown formatting like **bold** or # headers around
the section labels. Each label is at the start of a line, followed by
a colon and a space, then the content. Like this:

TITLE: 3-6 word title. No invented dates.
WHEN: A poetic time line, e.g. "Late August nights" or "March mornings".
      No invented date ranges.
BODY: {body_word_target} words of third-person narrative prose. Specific
      details from messages only. No invented psychology.
QUOTE: ONE actual TEXT message (not a shared reel caption) from the
      messages above. Quote it verbatim. Choose something specific to
      this chapter, appropriate for public display. Avoid generic
      phrases ("good night", "love you") and profanity.
QUOTE_BY: The person who wrote the quoted message.
ILLUSTRATION: A 15-word visual scene description for an illustrator.
      Describe a SCENE, not the people. e.g. "A dim apartment late at
      night, a phone glowing on a bedside table."
"""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_highlights(msgs: list[Message]) -> str:
    """Format messages for the chapter prompt. Critically: tag media
    placeholders so the LLM doesn't attribute reel captions as dialogue."""
    lines = []
    for m in msgs:
        time_str = m.timestamp.strftime("%Y-%m-%d %H:%M")
        if m.kind == MessageKind.MEDIA_PLACEHOLDER:
            # Make it VERY clear this is shared content, not a typed message
            lines.append(
                f"[{time_str}] {m.sender} [SHARED MEDIA — caption follows, "
                f"NOT their own words]: {m.text}"
            )
        else:
            lines.append(f"[{time_str}] {m.sender}: {m.text}")
    return "\n".join(lines)


def _build_date_range_text(start: date, end: date) -> str:
    """Produce an unambiguous range string. For single-day chapters
    we say so explicitly so the LLM can't expand to a multi-day span."""
    if start == end:
        return f"a single day — {start.isoformat()} only (no other dates)"
    days = (end - start).days
    if days < 7:
        return f"{start.isoformat()} to {end.isoformat()} ({days + 1} days)"
    return f"{start.isoformat()} to {end.isoformat()}"


def _body_word_target(message_count: int) -> str:
    """Adaptive body length. Counts TEXT messages (after media-share
    filtering). Now that we pass full conversation context, longer
    chats can support longer chapters."""
    if message_count < 10:
        return "60-120"
    if message_count < 30:
        return "120-200"
    if message_count < 80:
        return "180-280"
    if message_count < 200:
        return "220-320"
    return "280-380"


def _build_entities_block(entities: dict[str, list[str]]) -> str:
    """Format the extracted entities for the prompt, only including
    sections that actually have content."""
    parts = []
    if entities["nicknames"]:
        parts.append(
            "Nicknames they use for each other (use these where natural): "
            + ", ".join(entities["nicknames"])
        )
    if entities["other_people"]:
        parts.append(
            "Other people mentioned in messages (weave in where supported): "
            + ", ".join(entities["other_people"])
        )
    if entities["proper_nouns"]:
        parts.append(
            "Other names referenced in messages: "
            + ", ".join(entities["proper_nouns"])
        )
    if not parts:
        return ""
    return "== ENTITIES MENTIONED IN MESSAGES ==\n" + "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing & quote validation
# ---------------------------------------------------------------------------

def _pick_fallback_quote(messages: list[Message]) -> tuple[str, str]:
    """Pick a clean fallback pull-quote from raw messages if the LLM's
    pick gets rejected. TEXT-only (no shared media), medium-length,
    not generic/inappropriate."""
    from . import content_filter
    candidates = [
        m for m in messages
        if m.kind == MessageKind.TEXT
        and 15 <= len(m.text) <= 140
        and content_filter.safe_for_display(m.text)
    ]
    if not candidates:
        return "", ""
    chosen = candidates[len(candidates) // 2]
    return chosen.text, chosen.sender


def _validate_quote_is_real_message(
    quote: str, chapter_messages: list[Message]
) -> bool:
    """Verify the LLM's pull-quote is text from an actual TEXT message
    in the chapter (not a media share, not invented). Match is fuzzy —
    we just check that a substantial fraction of the quote appears
    verbatim in some text message.
    """
    if not quote:
        return False
    quote_norm = re.sub(r"\s+", " ", quote.lower().strip()).strip('"\'')
    if len(quote_norm) < 5:
        return False
    for m in chapter_messages:
        if m.kind != MessageKind.TEXT:
            continue
        msg_norm = re.sub(r"\s+", " ", m.text.lower().strip())
        if quote_norm in msg_norm or msg_norm in quote_norm:
            return True
        # Allow ~80% character match too (LLM may slightly reformat)
        if len(quote_norm) > 20 and len(msg_norm) > 20:
            shorter, longer = sorted([quote_norm, msg_norm], key=len)
            if shorter in longer:
                return True
    return False


def _strip_markdown(text: str) -> str:
    """Remove markdown decorations the LLM may add to section headers.
    Llama-family models frequently bold their section headers
    (**TITLE:** instead of TITLE:) which breaks naive prefix matching.
    Strip them so the existing parser logic works."""
    # **bold** and __bold__
    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"\1", text)
    text = re.sub(r"__([^_\n]+?)__", r"\1", text)
    # *italic* and _italic_ — careful not to eat single * mid-word
    text = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"\1", text)
    text = re.sub(r"(?<![\w_])_([^_\n]+?)_(?![\w_])", r"\1", text)
    # Heading hashes (## TITLE → TITLE) and blockquote markers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    # Bullet markers some models add to sections (- TITLE: → TITLE:)
    text = re.sub(r"^[-•]\s+(?=[A-Z]{3,}\s*:)", "", text, flags=re.MULTILINE)
    return text


def _smart_body_fallback(raw_response: str, sections: dict) -> str:
    """If BODY: wasn't found but the LLM clearly wrote prose, try to
    find it heuristically. Picks the longest paragraph that isn't a
    section header. Better than showing 'Chapter generation incomplete.'"""
    paragraphs = [p.strip() for p in raw_response.split("\n\n") if p.strip()]
    # Reject paragraphs that look like section headers
    candidates = [
        p for p in paragraphs
        if not re.match(r"^\s*(TITLE|WHEN|BODY|QUOTE|QUOTE_BY|ILLUSTRATION)\s*:",
                        p, re.IGNORECASE)
        and len(p) > 80
        and "section" not in p.lower()[:30]
    ]
    if not candidates:
        return ""
    # Pick the longest — it's almost certainly the body prose
    return max(candidates, key=len)


def _parse_chapter_response(
    text: str, index: int, fallback_when: str,
    chapter_messages: list[Message] | None = None,
) -> Chapter:
    """Parse the structured response. Validates the pull-quote against
    the source so an invented or media-share quote can't slip through."""
    from . import content_filter

    # Pre-pass: strip markdown decorations so headers match regardless
    # of whether the LLM bolded them.
    text = _strip_markdown(text)

    sections: dict[str, str] = {}
    current_key = None
    buffer: list[str] = []

    for line in text.splitlines():
        m = None
        for key in ["TITLE", "WHEN", "BODY", "QUOTE_BY", "QUOTE", "ILLUSTRATION"]:
            prefix = f"{key}:"
            if line.strip().upper().startswith(prefix):
                if current_key:
                    sections[current_key] = "\n".join(buffer).strip()
                current_key = key
                buffer = [line.split(":", 1)[1].strip()]
                m = key
                break
        if m is None and current_key:
            buffer.append(line)

    if current_key:
        sections[current_key] = "\n".join(buffer).strip()

    # If BODY came back empty or missing, try the smart fallback before
    # giving up. Many parse failures are recoverable — the LLM wrote
    # the prose, our header detection just missed.
    body = sections.get("BODY", "").strip()
    if not body or body.lower().startswith("("):
        recovered = _smart_body_fallback(text, sections)
        if recovered:
            body = recovered

    # Validate pull-quote: must be safe AND must be from an actual TEXT
    # message in the chapter (not a media share, not invented).
    quote = sections.get("QUOTE", "").strip('"\'')
    quote_by = sections.get("QUOTE_BY", "")
    needs_fallback = (
        not quote
        or not content_filter.safe_for_display(quote)
        or (chapter_messages and not _validate_quote_is_real_message(
            quote, chapter_messages))
    )
    if needs_fallback and chapter_messages:
        fallback_q, fallback_by = _pick_fallback_quote(chapter_messages)
        quote, quote_by = fallback_q, fallback_by
    elif needs_fallback:
        quote, quote_by = "", ""

    return Chapter(
        index=index,
        title=sections.get("TITLE", f"Chapter {index}"),
        when=sections.get("WHEN", fallback_when),
        body=body or "(Chapter generation incomplete.)",
        pull_quote=quote,
        pull_quote_author=quote_by,
        illustration_prompt=sections.get("ILLUSTRATION", "two friends in a cozy room"),
    )


# ---------------------------------------------------------------------------
# Detailed commentary generation
# ---------------------------------------------------------------------------

SCENE_PROMPT = """Write this small slice of a private chat as a soft diary
entry. Treat the messages as the ONLY source of truth.

== PRIOR MEMORY ==
{memory}

== PEOPLE / REAL REFERENCES ==
{sender_list}
{entities_block}

== THIS SLICE OF MESSAGES ==
Slice {scene_index} of {scene_count}. Continue from prior memory; do not
restart as if this is a new conversation.

{source_block}

== MANDATORY COVERAGE ==
{coverage_block}

== STYLE ==
- Write like a cute, private diary page, not a report and not analysis.
- Use warm, human prose: "I remember...", "It felt like...", "By 03:17...".
- Start with the actual time range in a natural way, e.g. "Around 03:13...".
- Keep events in the same order as the exported chat timestamps.
- Cover the important lines in this slice, but do not list them mechanically.
- Mention the small turns: short replies, "hmm", repeated asks, refusals,
  nicknames, pauses, softening, and pushback.
- You may describe apparent tone only when the words support it. Use careful
  language such as "seemed", "read like", "came across as", or "as if".
- Do not call something playful/lighthearted unless the messages clearly show
  joking, laughter, or emoji. Firm refusals like "No means no" are not playful
  by default.
- Do not translate Hinglish/Hindi unless the meaning is obvious. If uncertain,
  keep the original phrase and explain what role it played in the exchange.
- When a rough English meaning is supplied, use it to help English readers,
  but keep the original phrase nearby for accuracy.
- Do not claim private thoughts as fact. Do not invent motives, backstory,
  dates, locations, or events.
- Quote or paraphrase concrete messages so the commentary stays grounded.
- If someone asks for a pic/photo or refuses with reasons, narrate that
  plainly and specifically.
- Do not start every slice with "The conversation began"; vary the continuation
  naturally.
- Do not mention message IDs, "mandatory coverage", "rough English", "source",
  "exported timestamps", or "this slice" in the story.

== OUTPUT FORMAT ==
STORY: 140-260 words of diary-like chronological prose.
MEMORY: 2-4 factual bullets to carry into the next slice. No invented emotion.
"""


def _split_for_commentary(
    messages: list[Message],
    max_text_messages: int = 8,
    max_chars: int = 1400,
    gap_minutes: int = 12,
) -> list[list[Message]]:
    """Split a chapter into small chronological source windows.

    Local 8GB Ollama setups do better with several modest calls than one
    giant prompt. The chunks are intentionally based on raw chronological
    flow, not "top highlights", so dense arguments do not get compressed
    away.
    """
    text_msgs = [m for m in messages if m.kind == MessageKind.TEXT]
    if not text_msgs:
        return []

    chunks: list[list[Message]] = []
    current: list[Message] = []
    current_chars = 0

    for msg in text_msgs:
        gap_break = False
        if current:
            gap = msg.timestamp - current[-1].timestamp
            gap_break = gap.total_seconds() > gap_minutes * 60 and len(current) >= 6

        size_break = (
            len(current) >= max_text_messages
            or (current_chars + len(msg.text) > max_chars and len(current) >= 8)
        )
        if current and (gap_break or size_break):
            chunks.append(current)
            current = []
            current_chars = 0

        current.append(msg)
        current_chars += len(msg.text)

    if current:
        chunks.append(current)
    return chunks


def _rough_english_hint(text: str) -> str:
    """Small deterministic glosses for common Hinglish in this app's chats.

    This is intentionally modest. The LLM can improve the wording, but these
    hints keep English-reader meaning from disappearing when the local model
    is uncertain.
    """
    lower = text.lower()
    hints: list[str] = []
    phrase_hints = [
        ("no means no", "no means no"),
        ("situation alag", "the situation is different"),
        ("didi", "elder sister"),
        ("dekh legi", "will see it"),
        ("purane dekh", "look at old ones"),
        ("mt bhej", "do not send"),
        ("mat bhej", "do not send"),
        ("nhi bhej", "will not send"),
        ("nahi bhej", "will not send"),
        ("pic", "picture"),
        ("photo", "photo"),
        ("bhej", "send"),
        ("kyun nhi", "why not"),
        ("batao pehle", "tell me first"),
        ("phir mai samjh", "then I will understand"),
        ("nahi", "no"),
        ("nhi", "no"),
        ("samjho", "please understand"),
        ("mere sath", "with me"),
        ("soyi bhi nhi", "has not slept yet"),
        ("dil nhi", "does not feel like it"),
        ("phone", "phone"),
        ("roo", "crying"),
        ("ro rahi", "crying"),
        ("good night", "good night"),
        ("love you", "love you"),
    ]
    for needle, hint in phrase_hints:
        if needle in lower and hint not in hints:
            hints.append(hint)
    if ("mt bhej" in lower or "mat bhej" in lower) and "send" in hints:
        hints = [h for h in hints if h != "send"]
    if "7%" in text:
        hints.append("phone battery is at 7%")
    return "; ".join(hints[:5])


def _format_source_block(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        hint = _rough_english_hint(m.text)
        suffix = f" | English sense: {hint}" if hint else ""
        lines.append(
            f"[{m.timestamp.strftime('%H:%M')}] {m.sender}: {m.text}{suffix}"
        )
    return "\n".join(lines)


def _time_range(messages: list[Message]) -> str:
    if not messages:
        return ""
    return (
        f"{messages[0].timestamp.strftime('%H:%M')}-"
        f"{messages[-1].timestamp.strftime('%H:%M')}"
    )


def _coverage_block(messages: list[Message]) -> str:
    """Name the specific source lines that the LLM must not skip."""
    important_words = (
        "pic", "photo", "bhej", "bhejo", "didi", "situation", "phone",
        "7%", "nahi", "samjho", "no means no", "good night", "love you",
    )
    lines = []
    for m in messages:
        lower = m.text.lower()
        if any(word in lower for word in important_words):
            hint = _rough_english_hint(m.text)
            suffix = f" (English sense: {hint})" if hint else ""
            lines.append(
                f"- At {m.timestamp.strftime('%H:%M')}: {m.text}{suffix}"
            )
    return "\n".join(lines) if lines else "- Cover each message in order."


def _required_terms(messages: list[Message]) -> set[str]:
    text = " ".join(m.text.lower() for m in messages)
    terms: set[str] = set()
    if "didi" in text:
        terms.add("didi")
    if "situation" in text:
        terms.add("situation")
    if "pic" in text or "photo" in text:
        terms.add("photo")
    if "bhej" in text:
        terms.add("send")
    if "phone" in text or "7%" in text:
        terms.add("phone")
    if "nahi" in text or "no means no" in text:
        terms.add("refusal")
    return terms


def _story_has_term(story: str, term: str) -> bool:
    lower = story.lower()
    aliases = {
        "photo": ("photo", "picture", "pic"),
        "send": ("send", "sent", "bhej", "bhejo"),
        "phone": ("phone", "battery", "7%"),
        "refusal": ("refus", "no means no", "nahi", "no"),
        "didi": ("didi", "sister"),
        "situation": ("situation",),
    }
    return any(alias in lower for alias in aliases.get(term, (term,)))


def _deterministic_timeline_beat(messages: list[Message]) -> str:
    """Guaranteed grounded fallback when the model skips important lines."""
    if not messages:
        return ""
    parts = []
    for m in messages:
        hint = _rough_english_hint(m.text)
        if hint:
            parts.append(
            f"At {m.timestamp.strftime('%H:%M')}, {m.sender}'s line "
            f"\"{m.text}\" carried the sense of {hint}."
        )
        else:
            parts.append(
                f"At {m.timestamp.strftime('%H:%M')}, {m.sender} wrote "
                f"\"{m.text}\"."
            )
    return (
        f"I do not want to lose the exact order of this part: " + " ".join(parts)
    )


def _ensure_timeline_grounding(story: str, messages: list[Message]) -> str:
    """Keep a hook point for future validation without adding debug prose.

    Earlier builds appended deterministic source-line dumps when the model
    skipped a key detail. That protected facts but made the book feel like a
    log. The current smaller scene windows and mandatory coverage prompt keep
    accuracy without sacrificing diary voice.
    """
    return story


def _format_source_block_legacy(messages: list[Message]) -> str:
    return "\n".join(
        f"[{m.timestamp.strftime('%H:%M')}] {m.sender}: {m.text}"
        for m in messages
    )


def _parse_scene_response(text: str) -> tuple[str, str]:
    text = _strip_markdown(text)
    story = ""
    memory = ""
    current_key = None
    buffer: list[str] = []

    for line in text.splitlines():
        upper = line.strip().upper()
        if upper.startswith("STORY:") or upper.startswith("MEMORY:"):
            if current_key == "STORY":
                story = "\n".join(buffer).strip()
            elif current_key == "MEMORY":
                memory = "\n".join(buffer).strip()
            current_key = "STORY" if upper.startswith("STORY:") else "MEMORY"
            buffer = [line.split(":", 1)[1].strip()]
        elif current_key:
            buffer.append(line)

    if current_key == "STORY":
        story = "\n".join(buffer).strip()
    elif current_key == "MEMORY":
        memory = "\n".join(buffer).strip()

    if not story:
        story = _smart_body_fallback(text, {}) or text.strip()
    return story.strip(), memory.strip()


# Stopwords used when deriving a chapter title/illustration from raw words.
_TITLE_STOPWORDS = set("""
a an the and or but if then so for to of in on at by with from as is am are
was were be been being have has had do does did will would could should can
i me my you your we us our he she it they them this that these those not no
yes ok okay yeah ya just like get got go going gonna want need think know
say said u ur n r y idk btw omg pls plz really very also what when how why
who where which here there now then today tomorrow yesterday lol haha hey hi
""".split())


def _distinctive_words(messages: list[Message], limit: int = 3) -> list[str]:
    """Most frequent content words across a chapter's text messages."""
    from collections import Counter
    counter: Counter = Counter()
    for m in messages:
        if m.kind != MessageKind.TEXT:
            continue
        for word in re.findall(r"[a-zA-Z']{3,}", m.text.lower()):
            if word not in _TITLE_STOPWORDS:
                counter[word] += 1
    return [w for w, _ in counter.most_common(limit)]


def _chapter_title_from_messages(index: int, messages: list[Message]) -> str:
    """Derive a short, content-grounded title from the chapter's own words.

    Avoids inventing narrative the messages don't support, and avoids the
    old hardcoded titles that were tuned to one specific chat.
    """
    words = _distinctive_words(messages, limit=2)
    if words:
        return " & ".join(w.capitalize() for w in words)
    if messages:
        return messages[0].timestamp.strftime("%B %Y")
    return f"Chapter {index}"


def _illustration_prompt_from_messages(messages: list[Message]) -> str:
    """A per-chapter clipart-matching prompt. Uses the chapter's hour and
    distinctive words so different chapters pick different illustrations
    instead of all sharing one hardcoded prompt."""
    words = _distinctive_words(messages, limit=4)
    hour = messages[0].timestamp.hour if messages else 21
    if hour < 6 or hour >= 21:
        scene = "a phone glowing in a dim room late at night"
    elif hour < 12:
        scene = "morning light, a warm cup beside a phone"
    else:
        scene = "a cozy afternoon, messages on a screen"
    extra = (", themes: " + ", ".join(words)) if words else ""
    return f"{scene}{extra}"


def _chapter_when_from_messages(start: date, end: date, messages: list[Message]) -> str:
    if not messages:
        return start.strftime("%B %d, %Y")
    first = messages[0].timestamp.strftime("%H:%M")
    last = messages[-1].timestamp.strftime("%H:%M")
    if start == end:
        return f"{start.strftime('%B %d, %Y')}, {first}-{last}"
    return f"{start.strftime('%B %d, %Y')} to {end.strftime('%B %d, %Y')}"


def _pick_commentary_quote(messages: list[Message]) -> tuple[str, str]:
    """Prefer a quote that explains the chapter's real turning point."""
    from . import content_filter

    keyword_weights = [
        ("situation", 5), ("didi", 5), ("photo", 4), ("pic", 4),
        ("bhej", 4), ("phone", 3), ("7%", 3), ("no means no", 5),
        ("samjho", 3), ("nahi", 2),
    ]
    best: tuple[int, int, Message] | None = None
    for idx, msg in enumerate(messages):
        if msg.kind != MessageKind.TEXT:
            continue
        if not 5 <= len(msg.text) <= 180:
            continue
        if not content_filter.safe_for_display(msg.text):
            continue
        lower = msg.text.lower()
        score = sum(weight for word, weight in keyword_weights if word in lower)
        score += min(len(msg.text), 80) // 20
        candidate = (score, -idx, msg)
        if best is None or candidate > best:
            best = candidate
    if best:
        msg = best[2]
        return msg.text, msg.sender
    return _pick_fallback_quote(messages)


async def _generate_commentary_body(
    chapter_messages: list[Message],
    sender_names: set[str],
    entities: dict[str, list[str]],
) -> str:
    chunks = _split_for_commentary(chapter_messages)
    if not chunks:
        return "(No text messages available for this chapter.)"

    memory = "No prior memory for this chapter yet."
    sections: list[str] = []
    entities_block = _build_entities_block(entities) or "(none detected)"

    scene_count = len(chunks)
    for scene_index, scene_messages in enumerate(chunks, 1):
        prompt = SCENE_PROMPT.format(
            memory=memory,
            sender_list=", ".join(sorted(sender_names)),
            entities_block=entities_block,
            scene_index=scene_index,
            scene_count=scene_count,
            source_block=_format_source_block(scene_messages),
            coverage_block=_coverage_block(scene_messages),
        )
        response = await llm.complete(
            [
                {"role": "system", "content": (
                    "You are a grounded diary writer. You make chat exports "
                    "feel human and tender without inventing facts. You can "
                    "describe apparent tone only from the actual words and "
                    "message order."
                )},
                {"role": "user", "content": prompt},
            ],
            model_size="strong",
            temperature=0.25,
            max_tokens=650,
        )
        story, next_memory = _parse_scene_response(response)
        if story:
            sections.append(_ensure_timeline_grounding(story, scene_messages))
        if next_memory:
            memory = next_memory[:900]

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Two-Pass Generation System
# ---------------------------------------------------------------------------

def _chunk_with_overlap(
    messages: list[Message],
    chunk_size: int = 25,
    overlap: int = 5,
) -> list[list[Message]]:
    """Split messages into chunks with sliding window overlap.

    For example, with chunk_size=25 and overlap=5:
    - Chunk 0: messages[0:25]
    - Chunk 1: messages[20:45]
    - Chunk 2: messages[40:65]
    - etc.

    This maintains narrative continuity while keeping context windows manageable.
    """
    if not messages:
        return []

    text_msgs = [m for m in messages if m.kind == MessageKind.TEXT]
    if not text_msgs:
        return []

    if len(text_msgs) <= chunk_size:
        return [text_msgs]

    chunks: list[list[Message]] = []
    step = chunk_size - overlap

    for i in range(0, len(text_msgs), step):
        chunk = text_msgs[i : i + chunk_size]
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(text_msgs):
            break

    return chunks


EMOTIONAL_EXTRACTION_PROMPT = """Analyze this chat log between two people. Extract ONLY what is explicitly shown.

== MESSAGES ==
{formatted_messages}

== TASK ==
Extract two things:
1. EVENTS: Literal facts (what each person said, what was decided, what changed)
2. TENSIONS: Emotional undercurrents (longing, conflict, affection, sadness, resolution)

DO NOT invent emotions. DO NOT interpret tone beyond what the actual words support.
If a message says "ok", do not infer frustration unless context clearly shows it.

OUTPUT FORMAT (parse as JSON):
{{
    "events": [
        "statement of concrete fact",
        "another fact from the messages"
    ],
    "tensions": [
        "identified emotional thread",
        "another tension"
    ]
}}
"""


STORY_GENERATION_WITH_EMOTIONAL_CONTEXT = """You are a romantic novelist writing a book about a real chat relationship.
Write strictly in THIRD PERSON. Do not use 'I', 'me', 'my', 'we', 'us', or 'our'.
Focus on the feelings of longing, sadness, and affection revealed in the conversation.
Do not mention battery percentages, technical details, or timestamps in the prose.

== EMOTIONAL ARC ==
{emotional_context}

== PEOPLE ==
{sender_list}

{entities_block}

== MESSAGES (your source of truth) ==
{formatted_messages}

== EXAMPLE ==
Bad (first-person, invented emotions):
"I felt like she was upset with me. Maybe I should have called sooner.
Everything between us felt fragile lately."

Good (third-person, grounded, specific):
"When he asked how her day was, her response came slowly—not anger, but something
closer to weariness. The pattern was becoming clear: his late hours at work kept
pushing against her need to feel prioritized. In the silence between messages,
a question lingered unasked: did the work matter more?"

== OUTPUT ==
Write 150-280 words in third person. Narrate what the messages show, not what
you infer. Be specific about what was said, how the other person replied, and what
tensions emerged. Focus on feelings revealed through actions and words, not
inventions.
"""


async def _extract_emotional_arc(
    chunk: list[Message],
) -> dict:
    """Call LLM to extract emotional arc from a message chunk."""
    formatted = _format_highlights(chunk)
    prompt = EMOTIONAL_EXTRACTION_PROMPT.format(formatted_messages=formatted)

    response = await llm.complete(
        [
            {
                "role": "system",
                "content": "You are a precise analyst. Extract facts and emotions from chat logs. Respond in valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        model_size="strong",
        temperature=0.1,
        max_tokens=500,
    )

    return _loads_json_lenient(response)


def _loads_json_lenient(response: str) -> dict:
    """Parse a JSON object out of an LLM response that may be wrapped in
    markdown code fences or surrounded by prose. Local models (Llama 3.1
    8B) frequently do this, and strict json.loads would silently return
    empty events/tensions, gutting the emotional grounding."""
    if not response:
        return {"events": [], "tensions": []}
    text = response.strip()
    # Strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Last resort: grab the first {...} block
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass
    return {"events": [], "tensions": []}


def _merge_emotional_arcs(arcs: list[dict]) -> str:
    """Combine multiple emotional extraction results into a cohesive summary."""
    all_events = []
    all_tensions = []

    for arc in arcs:
        if isinstance(arc.get("events"), list):
            all_events.extend(arc["events"])
        if isinstance(arc.get("tensions"), list):
            all_tensions.extend(arc["tensions"])

    events_text = "\n".join(f"- {e}" for e in all_events[:10]) if all_events else "(none detected)"
    tensions_text = "\n".join(f"- {t}" for t in all_tensions[:10]) if all_tensions else "(none detected)"

    return f"EVENTS FROM THIS CHAPTER:\n{events_text}\n\nEMOTIONAL UNDERCURRENTS:\n{tensions_text}"


async def _generate_chapter_with_two_pass(
    index: int,
    start_date: date,
    end_date: date,
    chapter_messages: list[Message],
    month_summaries: dict[date, str],
    arc_context: str,
) -> Chapter:
    """Two-pass chapter generation: emotional extraction → story writing.

    Pass 1: Extract emotional arc to guide story generation
    Pass 2: Write the chapter in third person based on emotional context
    """
    sender_names = {m.sender for m in chapter_messages}
    entities = _extract_entities(chapter_messages, sender_names)

    text_msgs = [m for m in chapter_messages if m.kind == MessageKind.TEXT]

    if not text_msgs:
        return Chapter(
            index=index,
            title=f"Chapter {index}",
            when=_chapter_when_from_messages(start_date, end_date, chapter_messages),
            body="(No messages available for this chapter)",
            pull_quote="",
            pull_quote_author="",
            illustration_prompt="A phone glowing in a dim room",
        )

    # Pass 1: Extract emotional arc from 25-message chunks with 5-message overlap
    chunks = _chunk_with_overlap(text_msgs, chunk_size=25, overlap=5)
    emotional_arcs: list[dict] = []

    for chunk in chunks:
        arc = await _extract_emotional_arc(chunk)
        emotional_arcs.append(arc)

    emotional_summary = _merge_emotional_arcs(emotional_arcs)

    # Pass 2: Generate story based on emotional context
    body = await _generate_story_from_emotional_arc(
        chapter_messages=chapter_messages,
        sender_names=sender_names,
        entities=entities,
        emotional_context=emotional_summary,
    )

    pull_quote, pull_quote_author = _pick_commentary_quote(chapter_messages)
    return Chapter(
        index=index,
        title=_chapter_title_from_messages(index, chapter_messages),
        when=_chapter_when_from_messages(start_date, end_date, chapter_messages),
        body=body,
        pull_quote=pull_quote,
        pull_quote_author=pull_quote_author,
        illustration_prompt=_illustration_prompt_from_messages(chapter_messages),
    )


async def _generate_story_from_emotional_arc(
    chapter_messages: list[Message],
    sender_names: set[str],
    entities: dict[str, list[str]],
    emotional_context: str,
) -> str:
    """Generate the story body using Pass 1 emotional arc as context."""
    chunks = _split_for_commentary(chapter_messages)
    if not chunks:
        return "(No text messages available for this chapter.)"

    sections: list[str] = []
    entities_block = _build_entities_block(entities) or "(none detected)"

    scene_count = len(chunks)
    memory = "No prior memory for this chapter yet."

    for scene_index, scene_messages in enumerate(chunks, 1):
        formatted_msgs = _format_highlights(scene_messages)

        prompt = STORY_GENERATION_WITH_EMOTIONAL_CONTEXT.format(
            emotional_context=emotional_context,
            sender_list=", ".join(sorted(sender_names)),
            entities_block=entities_block,
            formatted_messages=formatted_msgs,
        )

        response = await llm.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a romantic novelist writing in third person. "
                        "Do not use first-person perspective (I, me, we, us, our). "
                        "Focus on the feelings of longing, sadness, and affection "
                        "revealed in the conversation. Write based on what the messages show, "
                        "not on invented psychology."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model_size="strong",
            temperature=0.25,
            max_tokens=800,
        )

        if response:
            sections.append(response)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def generate_chapter(
    index: int,
    start_date: date,
    end_date: date,
    chapter_messages: list[Message],
    month_summaries: dict[date, str],
    arc_context: str,
) -> Chapter:
    return await _generate_chapter_with_two_pass(
        index=index,
        start_date=start_date,
        end_date=end_date,
        chapter_messages=chapter_messages,
        month_summaries=month_summaries,
        arc_context=arc_context,
    )
