"""Aggressive suppression of export and media noise before meaning extraction."""

from __future__ import annotations

import re

from ...models import Message, MessageKind


URL_RE = re.compile(r"(?:https?://|www\.)\S+|(?:instagram|youtu\.?be|youtube|spotify)\.com/\S*", re.I)
FORWARDED_RE = re.compile(
    r"\b(?:forwarded|sent an attachment|attachment|shared a (?:reel|post|video)|"
    r"media omitted|image omitted|video omitted|document omitted|sticker omitted)\b",
    re.I,
)
LOW_SEMANTIC_TOKENS = {
    "http", "https", "www", "com", "instagram", "reel", "reels", "forwarded",
    "attachment", "attached", "media", "omitted", "video", "image", "document",
    "shared", "post", "utm", "ref", "watch", "status", "message", "deleted",
}
FILLER_TOKENS = {
    "ok", "okay", "hmm", "hm", "kk", "gm", "gn", "hey", "hello", "yeah",
    "yes", "no", "acha", "accha", "haan", "nahi",
}


def suppress_noise(text: str) -> str:
    """Remove transport/media metadata while retaining actual human words."""
    value = URL_RE.sub(" ", text.casefold())
    value = FORWARDED_RE.sub(" ", value)
    value = re.sub(r"[_#@/=:?&.%+-]+", " ", value)
    tokens = [
        token for token in re.findall(r"[a-z']{2,}", value)
        if token not in LOW_SEMANTIC_TOKENS
    ]
    return " ".join(tokens)


def semantic_tokens(text: str) -> list[str]:
    return [
        token for token in suppress_noise(text).split()
        if token not in FILLER_TOKENS
    ]


def is_semantic_message(message: Message) -> bool:
    if message.kind != MessageKind.TEXT:
        return False
    cleaned = suppress_noise(message.normalized_text or message.text)
    tokens = semantic_tokens(cleaned)
    return bool(tokens) and not (len(tokens) <= 2 and all(token in FILLER_TOKENS for token in cleaned.split()))
