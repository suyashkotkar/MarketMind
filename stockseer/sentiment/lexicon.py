"""Compact finance-domain sentiment lexicon.

Modelled on the Loughran–McDonald approach: general-purpose lexicons mislabel
financial text ("liability", "tax", "cost" are not negative in a 10-K). Weights
are in [-1, 1]. Kept in-repo so scoring is deterministic and needs no download.
"""
from __future__ import annotations

POSITIVE: dict[str, float] = {
    "beat": 0.8, "beats": 0.8, "outperform": 0.8, "outperformed": 0.8,
    "upgrade": 0.9, "upgraded": 0.9, "raise": 0.6, "raises": 0.6, "raised": 0.6,
    "record": 0.6, "surge": 0.8, "surged": 0.8, "soar": 0.9, "soared": 0.9,
    "rally": 0.7, "rallied": 0.7, "jump": 0.6, "jumped": 0.6, "gain": 0.5,
    "gains": 0.5, "growth": 0.5, "profit": 0.6, "profitable": 0.7,
    "strong": 0.6, "strength": 0.6, "robust": 0.6, "solid": 0.5,
    "expansion": 0.5, "expanding": 0.5, "buyback": 0.6, "dividend": 0.4,
    "wins": 0.6, "win": 0.6, "won": 0.6, "approval": 0.6, "approved": 0.6,
    "breakthrough": 0.8, "optimistic": 0.6, "bullish": 0.9, "buy": 0.5,
    "accelerate": 0.5, "accelerating": 0.5, "exceeded": 0.8, "exceeds": 0.8,
    "improve": 0.5, "improved": 0.5, "improving": 0.5, "rebound": 0.6,
    "milestone": 0.5, "demand": 0.3, "efficiency": 0.4, "innovative": 0.4,
    "partnership": 0.4, "expands": 0.5, "momentum": 0.5, "upside": 0.7,
}

NEGATIVE: dict[str, float] = {
    "miss": -0.8, "misses": -0.8, "missed": -0.8, "downgrade": -0.9,
    "downgraded": -0.9, "cut": -0.6, "cuts": -0.6, "slash": -0.8,
    "slashed": -0.8, "plunge": -0.9, "plunged": -0.9, "tumble": -0.8,
    "tumbled": -0.8, "slump": -0.7, "slumped": -0.7, "fall": -0.5,
    "fell": -0.5, "drop": -0.5, "dropped": -0.5, "decline": -0.6,
    "declined": -0.6, "loss": -0.7, "losses": -0.7, "weak": -0.6,
    "weakness": -0.6, "warning": -0.7, "warns": -0.7, "warned": -0.7,
    "lawsuit": -0.7, "probe": -0.7, "investigation": -0.7, "fraud": -1.0,
    "recall": -0.7, "bankruptcy": -1.0, "default": -0.9, "layoff": -0.7,
    "layoffs": -0.7, "restructuring": -0.4, "bearish": -0.9, "sell": -0.5,
    "risk": -0.3, "risks": -0.3, "concern": -0.5, "concerns": -0.5,
    "delay": -0.5, "delayed": -0.5, "halt": -0.6, "halted": -0.6,
    "shortfall": -0.8, "headwind": -0.6, "headwinds": -0.6, "pressure": -0.4,
    "disappointing": -0.8, "underperform": -0.8, "volatile": -0.3,
    "downside": -0.7, "sued": -0.7, "penalty": -0.6, "fine": -0.5,
    "resign": -0.5, "resigned": -0.5, "scandal": -0.9, "subpoena": -0.8,
}

# Words that flip the polarity of the next few tokens.
NEGATORS = {"not", "no", "never", "without", "fails", "fail", "failed",
            "unable", "less", "lacks", "lack", "cannot", "cant", "didnt",
            "doesnt", "isnt", "wasnt", "wont", "hardly", "barely"}

# Words that scale the next token's weight.
INTENSIFIERS = {"very": 1.5, "sharply": 1.5, "significantly": 1.4, "strongly": 1.4,
                "slightly": 0.6, "marginally": 0.6, "somewhat": 0.7,
                "massively": 1.8, "record": 1.3, "modestly": 0.7}

LEXICON: dict[str, float] = {**POSITIVE, **NEGATIVE}
