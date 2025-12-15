import re

# Map canonical relation -> list of alias patterns (longest first!)
RELATION_PATTERNS = {
    "left of": [
        r"\bto\s+the\s+left\s+of\b",
        r"\bon\s+the\s+left\s+of\b",
        r"\bleft\s+of\b",
        r"\bleft\b"  # fallback, but keep last
    ],
    "right of": [
        r"\bto\s+the\s+right\s+of\b",
        r"\bon\s+the\s+right\s+of\b",
        r"\bright\s+of\b",
        r"\bright\b"
    ],
    "above": [
        r"\bon\s+top\s+of\b",   # treat as 'above' semantically, or keep separate if you want
        r"\babove\b",
        r"\bover\b"
    ],
    "below": [
        r"\bbelow\b",
        r"\bunder\b",
        r"\bbeneath\b"
    ],
    "on": [
        r"\bon(?:to)?\b"  # matches 'on' and 'onto'
    ],
    "near": [
        r"\bnext\s+to\b",
        r"\badjacent\s+to\b",
        r"\bclose\s+to\b",
        r"\bnear\b",
        r"\bbeside\b"
    ],
    # Optional: keep 'top'/'bottom' as synonyms if these appear as short forms
    "top": [r"\btop\b"],
    "bottom": [r"\bbottom\b"],
}


# Small connector/stop-words to trim around object phrases
_CONNECTORS = re.compile(r"^(of|to|the|a|an)\b\s*|\s*\b(of|to|the|a|an)$", re.IGNORECASE)


def get_compiled_relation():
    # Precompile with case-insensitive flag and keep (canonical, regex, length) for longest-first sorting
    _COMPILED_RELATIONS = []
    for canonical, pats in RELATION_PATTERNS.items():
        for pat in pats:
            rx = re.compile(pat, flags=re.IGNORECASE)
            # approximate "length" by pattern string length to prioritize longer phrases
            _COMPILED_RELATIONS.append((canonical, rx, len(pat)))

    # Sort descending so longer/stricter patterns win
    _COMPILED_RELATIONS.sort(key=lambda x: x[2], reverse=True)

    return _COMPILED_RELATIONS

_COMPILED_RELATIONS = get_compiled_relation()

SKIP_KEYWORDS = {"left", "right", "top", "bottom", "front", "back", "center", "middle", "side", "background", "foreground"}