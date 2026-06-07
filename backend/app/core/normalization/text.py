"""Small, explainable text normalization for code-mixed Indian chats."""

from __future__ import annotations

import re
import unicodedata

from ...models import Message, MessageKind


PHRASE_ALIASES: dict[str, str] = {
    "i luv u": "love you",
    "luv u": "love you",
    "miss u": "miss you",
    "proud of u": "proud of you",
    "thnku": "thank you",
    "thanku": "thank you",
    "sry": "sorry",
    "plz": "please",
    "dont leave": "do not leave",
    "don't leave": "do not leave",
    "pyaar": "love",
    "pyar": "love",
    "yaad aa rahi": "miss you",
    "yaad aa raha": "miss you",
    "mujhe darr": "i am scared",
    "dar lag": "scared",
    "main hu na": "i am here for you",
    "mai hu na": "i am here for you",
    "tension mat lo": "do not worry",
    "chinta mat karo": "do not worry",
    "maaf kar": "sorry",
    "galti ho gayi": "sorry",
    "प्यार": "love",
    "याद आ रही": "miss you",
    "याद आ रहा": "miss you",
    "माफ़": "sorry",
    "मैं हूँ ना": "i am here for you",
    "nuvvu ante istam": "love you",
    "ninnu premistunna": "love you",
    "gurthostunnav": "miss you",
    "badha ga undi": "sad",
    "nenu unna": "i am here for you",
    "క్షమించు": "sorry",
    "నిన్ను ప్రేమిస్తున్నాను": "love you",
    "గుర్తొస్తున్నావు": "miss you",
    "బాధగా ఉంది": "sad",
}


def normalize_message_text(text: str) -> str:
    """Return a scoring-oriented form while preserving original message text."""
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    for phrase, replacement in PHRASE_ALIASES.items():
        value = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", replacement, value)
    return value


def normalize_messages(messages: list[Message]) -> list[Message]:
    """Populate `normalized_text` in place for backward-compatible models."""
    for message in messages:
        if message.kind == MessageKind.TEXT:
            message.normalized_text = normalize_message_text(message.text)
        else:
            message.normalized_text = message.text.casefold().strip()
    return messages
