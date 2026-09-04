"""Financial sentiment scoring.

Default: a fast, dependency-free lexicon scorer with negation + intensifier
handling. If `vaderSentiment` is installed we blend it in (it handles general
English better); the finance lexicon always dominates so domain terms win.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from .lexicon import INTENSIFIERS, LEXICON, NEGATORS

_TOKEN_RE = re.compile(r"[a-z']+")
NEGATION_WINDOW = 3


@dataclass
class SentimentResult:
    score: float          # -1 .. 1
    label: str            # positive | neutral | negative
    hits: int             # number of lexicon matches

    def as_dict(self) -> dict:
        return {"score": round(self.score, 4), "label": self.label, "hits": self.hits}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower().replace("’", "'"))


@lru_cache(maxsize=1)
def _vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def label_for(score: float, band: float = 0.05) -> str:
    if score > band:
        return "positive"
    if score < -band:
        return "negative"
    return "neutral"


def score_text(text: str) -> SentimentResult:
    tokens = _tokenize(text)
    if not tokens:
        return SentimentResult(0.0, "neutral", 0)

    total, hits = 0.0, 0
    for i, tok in enumerate(tokens):
        w = LEXICON.get(tok)
        if w is None:
            continue
        window = tokens[max(0, i - NEGATION_WINDOW):i]
        if any(t in NEGATORS for t in window):
            w = -w * 0.8
        for t in window[-2:]:
            if t in INTENSIFIERS:
                w *= INTENSIFIERS[t]
        total += w
        hits += 1

    if hits == 0:
        lex_score = 0.0
    else:
        # Normalise by sqrt(hits): long articles shouldn't automatically outrank
        # a punchy headline, but more evidence should still count for something.
        lex_score = max(-1.0, min(1.0, total / (hits ** 0.5) / 1.5))

    v = _vader()
    if v is not None:
        comp = v.polarity_scores(text)["compound"]
        lex_score = 0.7 * lex_score + 0.3 * comp if hits else comp * 0.5

    return SentimentResult(float(lex_score), label_for(lex_score), hits)


def score_texts(texts: Sequence[str] | Iterable[str]) -> list[SentimentResult]:
    return [score_text(t) for t in texts]
