from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...data.pipeline import load_news
from ...db.session import get_db
from ...sentiment.scorer import label_for, score_text
from ..schemas import NewsOut, SentimentOut

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


class ScoreRequest(BaseModel):
    texts: list[str]


@router.post("/score")
def score(req: ScoreRequest):
    """Score arbitrary text with the finance lexicon (useful for demos/tests)."""
    return [{"text": t, **score_text(t).as_dict()} for t in req.texts]


@router.get("/{symbol}", response_model=SentimentOut)
def sentiment(symbol: str, limit: int = Query(50, ge=1, le=500),
              db: Session = Depends(get_db)):
    news = load_news(db, symbol, limit=limit)
    if news.empty:
        raise HTTPException(404, f"no news stored for {symbol.upper()} — "
                                 "run ingestion with a news source configured")

    scores = news["sentiment"].dropna()
    mean = float(scores.mean()) if len(scores) else None
    counts = news["sentiment_label"].value_counts().to_dict()

    n = news.copy()
    n["day"] = pd.to_datetime(n["published_at"], utc=True).dt.date
    daily = (n.groupby("day")["sentiment"]
             .agg(["mean", "count"]).reset_index()
             .rename(columns={"mean": "sentiment", "count": "articles"}))
    daily_rows = [{"date": str(r["day"]),
                   "sentiment": None if pd.isna(r["sentiment"])
                   else round(float(r["sentiment"]), 4),
                   "articles": int(r["articles"])}
                  for r in daily.to_dict("records")]

    articles = [NewsOut(**r) for r in news.to_dict("records")]
    return SentimentOut(
        symbol=symbol.upper(), articles=articles,
        mean_sentiment=round(mean, 4) if mean is not None else None,
        label=label_for(mean) if mean is not None else None,
        counts={k: int(v) for k, v in counts.items()},
        daily=sorted(daily_rows, key=lambda r: r["date"]),
    )
