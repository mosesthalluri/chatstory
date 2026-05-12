"""
Content filtering for user-facing stats and quotes.

Two distinct concerns we handle separately:

  1. GENERIC phrases — common dialogue that isn't an "inside joke" even
     if it appears many times. "good night", "love you", "miss you" are
     boring, not signature. Filter from inside-joke candidates.

  2. INAPPROPRIATE content — profanity, slurs, sexual content that
     should NEVER appear on the user-facing Wrapped page or pull-quotes.
     Most chats between close people contain these; the book should not
     splash them on the showcase pages users will screenshot.

The lists are intentionally:
  - English + Hindi/Urdu transliteration (matches our Indian-market focus)
  - Conservative for the appropriate-content filter (false positives are
    fine; we'd rather miss showing an "inside joke" than display a slur)
  - Not exhaustive — they should be extended as we see real edge cases.
    Add new entries with a brief comment about what kind of input
    triggered it.

Used by:
  - pipeline/stats.py: to filter inside_jokes
  - pipeline/highlights.py: to filter candidate pull-quotes
"""

import re


# ---------------------------------------------------------------------------
# Generic dialogue — common phrases that are NOT inside jokes
# ---------------------------------------------------------------------------
# Each line is a phrase (lowercased, whitespace-normalized) we should NEVER
# present as an "inside joke". This list is matched against bigrams, so
# the entries are mostly 2-word phrases.

GENERIC_PHRASES = {
    # Greetings/farewells
    "good morning", "good night", "good evening", "good afternoon",
    "see you", "see ya", "talk later", "talk soon", "talk tomorrow",
    "bye bye", "good bye", "going bed", "going sleep", "sleep tight",
    "sleep well", "sweet dreams", "nighty night",
    # Affection (universal, not signature)
    "love you", "miss you", "missed you", "missing you", "love u",
    "miss u", "take care", "be careful", "be safe", "stay safe",
    "thinking you", "thinking about", "love love", "miss miss",
    "love too", "miss too",
    # Polite filler
    "thank you", "thanks lot", "thanks much", "no problem", "no worries",
    "sounds good", "looks good", "feeling good", "feeling bad",
    "feeling sad", "feeling tired", "feeling sleepy",
    # Universal Q&A
    "how are", "are you", "are u", "you doing", "u doing",
    "what doing", "what happened", "what going", "whats up", "whats wrong",
    "where are", "where you", "where u",
    "ok ok", "okay okay", "haa haa", "hmm hmm", "yeah yeah", "hnn hnn",
    # Common requests/responses
    "please please", "plz plz", "yes please", "no please", "baby please",
    "babe please", "tell me", "tell please", "listen me", "wait wait",
    # Hindi/Urdu transliterations (common in Indian chats)
    "kya kar", "kya hua", "kya kr", "kya re", "kar rahi", "kar rha", "kr rahi", "kr rha",
    "kahan ho", "kaha ho", "kaha hai", "kahan hai",
    "khana khaya", "khana kha",
    "ghar aa", "ghar aaja", "ghar gaye",
    "soo gaya", "soo gaye", "so gaye", "so gaya",
    "nind aa", "nind nhi", "nind nahi",
    "thik hai", "thik ho", "theek hai", "theek ho",
    "acha hai", "achha hai", "accha hai",
    "haa haa", "haan haan", "han han",
    "nahi nahi", "nhi nhi", "naa naa",
    "main hu", "mai hu", "tum ho", "tu hai",
    "pyaar hai", "pyaar kr", "pyar hai",
    "bhul gaya", "bhul gayi", "yaad aaya", "yaad aayi",
}


# ---------------------------------------------------------------------------
# Inappropriate content — must NEVER appear on user-facing stats/quotes
# ---------------------------------------------------------------------------
# A word in this list (or any token containing one of its substrings) makes
# the whole phrase ineligible for user-facing display. We're conservative:
# we'd rather drop a genuine inside joke than risk a slur on the showcase.
#
# This is for STATS DISPLAY ONLY. The underlying messages and chapter
# narrative are not affected — the LLM still sees the real conversation.

# Substrings that trigger filtering. Tested case-insensitively.
INAPPROPRIATE_SUBSTRINGS = {
    # English profanity (severe enough to block on Wrapped page)
    "fuck", "shit", "bitch", "asshole", "cunt", "dick", "pussy", "cock",
    "tits", "boobs", "horny", "naked", "nude", "porn", "sex", "sexual",
    "orgasm", "masturb", "jerk off", "blowjob", "handjob",
    # Slurs (block on user-facing)
    "nigger", "nigga", "faggot", "tranny", "retard", "spastic",
    "chink", "spic", "kike", "wetback",
    # Hindi/Urdu profanity transliterations (Indian-market specific)
    "madarchod", "maderchod", "mc bc", "behenchod", "bhenchod", "bsdk",
    "chutiya", "chutia", "chootiya", "gaand", "gandu", "lavde", "lavda",
    "lund", "land ka", "harami", "kamina", "kameena", "randi", "rand ki",
    "saala madarchod", "teri ma", "teri behen",
}


def contains_inappropriate(text: str) -> bool:
    """Return True if the text contains any inappropriate substring.
    Used to block phrases from user-facing display."""
    if not text:
        return False
    lowered = text.lower()
    return any(bad in lowered for bad in INAPPROPRIATE_SUBSTRINGS)


def is_generic_phrase(phrase: str) -> bool:
    """Return True if the phrase is generic dialogue that should not
    be flagged as an 'inside joke'."""
    if not phrase:
        return True
    normalized = re.sub(r"\s+", " ", phrase.lower().strip())
    return normalized in GENERIC_PHRASES


def safe_for_display(text: str) -> bool:
    """Master check: is this phrase safe to show on the Wrapped page,
    pull-quotes, or any other user-facing showcase surface?

    Returns False if the phrase is:
      - inappropriate (profanity/slurs)
      - a generic dialogue phrase (greetings, "love you", etc.)
      - empty or whitespace-only
    """
    if not text or not text.strip():
        return False
    if contains_inappropriate(text):
        return False
    if is_generic_phrase(text):
        return False
    return True


def has_distinctive_token(phrase: str, common_tokens: set[str]) -> bool:
    """A real inside joke usually contains at least one 'distinctive'
    token — a nickname, made-up word, or uncommon term — rather than
    being entirely composed of high-frequency dialogue words.

    `common_tokens` should be a set of the most common words across the
    chat (top 100 or so). If every token in the phrase is in that set,
    the phrase is probably generic. If at least one token is outside it,
    the phrase has signal.

    This is a heuristic — not perfect, but it filters out the worst
    "love you" / "good night" cases without needing a dictionary or LLM.
    """
    tokens = re.findall(r"[a-z']+", phrase.lower())
    if not tokens:
        return False
    return any(t not in common_tokens for t in tokens)
