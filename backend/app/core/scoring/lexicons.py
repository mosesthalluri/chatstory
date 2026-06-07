"""Editable signal vocabularies for relationship-session scoring."""

POSITIVE_SIGNALS: dict[str, set[str]] = {
    "affection": {"love you", "miss you", "pyaar", "pyar", "hug", "kiss", "babu", "baby"},
    "gratitude": {"thank you", "thanks", "grateful", "shukriya"},
    "support": {"i am here for you", "you got this", "take care", "proud of you", "do not worry"},
    "vulnerability": {"scared", "afraid", "i feel", "lonely", "can't sleep", "crying"},
    "apology": {"sorry", "forgive me", "maaf"},
    "reassurance": {"it is okay", "its okay", "everything will be okay", "trust me"},
    "future_planning": {"next time", "someday", "together", "meet soon", "marry", "future"},
    "openness": {"honestly", "truth is", "i need to tell", "feel about"},
}

NEGATIVE_SIGNALS: dict[str, set[str]] = {
    "fight": {"fight", "angry", "stop talking", "leave me", "hate", "shut up"},
    "sadness": {"sad", "cry", "hurt", "broken", "upset"},
    "grief": {"died", "death", "lost him", "lost her", "grief"},
    "insecurity": {"insecure", "not enough", "replace me", "do you even love"},
    "abandonment": {"leave me", "left me", "do not leave", "alone", "abandon", "ignored"},
    "betrayal": {"cheat", "lied", "betray", "trust broken"},
    "distance": {"don't care", "do not care", "over between us", "need space", "drifting"},
}

FILLER_TERMS = {
    "hi", "hey", "hello", "gm", "good morning", "gn", "good night",
    "ok", "okay", "k", "kk", "hmm", "hm", "yes", "no", "acha", "accha",
}

LOGISTICS_TERMS = {
    "where are you", "reached", "on my way", "call me", "send address",
    "location", "pickup", "payment", "otp", "meeting at", "class at",
}
