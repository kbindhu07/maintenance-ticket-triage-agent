"""Text cleaning and entity extraction for maintenance tickets."""
import re

# Common field abbreviations -> normalised words so the model sees one vocabulary.
ABBREVIATIONS = {
    r"\bwtg\b": "wind turbine",
    r"\binv\b": "inverter",
    r"\btemp\b": "temperature",
    r"\bvib\b": "vibration",
    r"\bgbx\b": "gearbox",
    r"\bcomm\b": "communication",
    r"\bcomms\b": "communication",
    r"\btrf\b": "transformer",
    r"\bxfmr\b": "transformer",
    r"\bpm\b": "preventive maintenance",
    r"\bhrs?\b": "hours",
    r"\bdc\b": "dc",
}

ASSET_ID_PATTERNS = [
    (r"\bWTG[-\s]?\d+\b", "wind_turbine"),
    (r"\binverter\s*#?\s*\d+\b", "inverter"),
    (r"\binv[-\s]?\d+\b", "inverter"),
    (r"\bblock\s*#?\s*\d+\b", "solar_block"),
    (r"\bT[-\s]?\d+\b(?=.*transformer)|\btransformer\s*#?\s*[A-Z]?\d+\b", "transformer"),
    (r"\bunit\s*#?\s*\d+\b", "unit"),
    (r"\bgate\s*#?\s*\d+\b", "gate"),
    (r"\bfeeder\s*#?\s*\d+\b", "feeder"),
]

ASSET_TYPE_HINTS = {
    "wind": ["turbine", "wtg", "blade", "pitch", "gearbox", "nacelle", "yaw", "rotor"],
    "solar": ["inverter", "pv", "module", "string", "block", "panel", "irradiance", "soiling"],
    "hydro": ["spillway", "penstock", "dam", "reservoir", "gate", "hydro", "francis", "pelton"],
    "grid": ["substation", "transformer", "feeder", "sldc", "curtailment", "dispatch", "busbar", "grid"],
}

DURATION_PATTERN = r"\b(?:last|for|since)\s+(?:the\s+)?(?:\w+\s+)?(\d+|one|two|three|four|five|six|several)?\s*(minutes?|hours?|days?|weeks?)\b"


def contains_term(term: str, text: str) -> bool:
    """Match a term at a word start, so 'trip' matches 'tripping' but 'fall' does not match 'shortfall'."""
    return re.search(r"\b" + re.escape(term), text) is not None


def find_terms(terms, text: str) -> list:
    return [t for t in terms if contains_term(t, text)]


def clean_text(text: str) -> str:
    """Lowercase, expand abbreviations, collapse whitespace. Used for the ML model."""
    t = (text or "").lower()
    for pattern, repl in ABBREVIATIONS.items():
        t = re.sub(pattern, repl, t)
    t = re.sub(r"[^a-z0-9\s\-#/.%]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_entities(text: str) -> dict:
    """Pull out structured facts the agent can reason about."""
    raw = text or ""
    lowered = clean_text(raw)
    asset_ids = []
    for pattern, kind in ASSET_ID_PATTERNS:
        for m in re.finditer(pattern, raw, flags=re.IGNORECASE):
            value = m.group(0).strip()
            if value and {"id": value, "kind": kind} not in asset_ids:
                asset_ids.append({"id": value, "kind": kind})

    hint_scores = {k: sum(1 for w in words if w in lowered) for k, words in ASSET_TYPE_HINTS.items()}
    best = max(hint_scores, key=hint_scores.get)
    inferred_asset_type = best if hint_scores[best] > 0 else None

    durations = [" ".join(x for x in m.groups() if x) for m in re.finditer(DURATION_PATTERN, lowered)]
    numbers = re.findall(r"\b\d+(?:\.\d+)?\s?(?:%|mw|kw|mwh|degc|°c|c|hz|kv|mm/s)\b", raw.lower())

    return {
        "asset_ids": asset_ids,
        "inferred_asset_type": inferred_asset_type,
        "asset_type_hint_scores": hint_scores,
        "durations": durations,
        "measurements": numbers,
        "clean_text": lowered,
    }
