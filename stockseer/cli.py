"""Command line entry point.

    python -m stockseer.cli init-db
    python -m stockseer.cli ingest --symbols AAPL,MSFT --period 5y
    python -m stockseer.cli train
    python -m stockseer.cli predict AAPL
    python -m stockseer.cli risk AAPL
    python -m stockseer.cli anomalies AAPL
    python -m stockseer.cli pipeline          # ingest + train + predict, end to end
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import settings
from .db.session import init_db, session_scope

log = logging.getLogger("stockseer.cli")


def _out(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _symbols(arg: str | None) -> list[str] | None:
    return [s.strip().upper() for s in arg.split(",")] if arg else None


def cmd_init_db(_args):
    init_db()
    _out({"status": "ok", "database": settings.database_url})


def cmd_ingest(args):
    from .data.pipeline import ingest_universe
    init_db()
    with session_scope() as db:
        reports = ingest_universe(db, _symbols(args.symbols), args.period,
                                  not args.no_news)
    _out([r.as_dict() for r in reports])


def cmd_train(args):
    from .api.services import analytics
    init_db()
    with session_scope() as db:
        res = analytics.train(db, _symbols(args.symbols), args.horizon,
                              args.model_type)
    _out(res)


def cmd_predict(args):
    from .api.services import analytics
    with session_scope() as db:
        if args.symbol:
            _out(analytics.predict_symbol(db, args.symbol))
        else:
            _out(analytics.predict_universe(db))


def cmd_risk(args):
    from .api.services import analytics
    with session_scope() as db:
        _out(analytics.risk_for(db, args.symbol))


def cmd_anomalies(args):
    from .api.services import analytics
    with session_scope() as db:
        _out(analytics.anomalies_for(db, args.symbol,
                                     lookback_days=args.lookback))


def cmd_compare(args):
    from .api.services import analytics
    with session_scope() as db:
        res = analytics.compare(db, _symbols(args.symbols))
    _out(res["rows"])


def cmd_pipeline(args):
    from .api.services import analytics
    from .data.pipeline import ingest_universe
    init_db()
    syms = _symbols(args.symbols)
    with session_scope() as db:
        reports = ingest_universe(db, syms, args.period, not args.no_news)
        ok = [r.symbol for r in reports if not r.error]
        log.info("ingested %d symbols", len(ok))
        train_res = analytics.train(db, None, args.horizon, args.model_type)
        preds = analytics.predict_universe(db)
    _out({
        "ingest": [r.as_dict() for r in reports],
        "training": {"version": train_res["version"],
                     "n_rows": train_res["n_rows"],
                     "out_of_fold": train_res["metrics"]["out_of_fold"],
                     "backtest": train_res["metrics"]["backtest"]},
        "predictions": preds,
    })


def cmd_serve(args):
    import uvicorn
    uvicorn.run("stockseer.api.main:app", host=args.host, port=args.port,
                reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stockseer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default=settings.log_level)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db").set_defaults(func=cmd_init_db)

    a = sub.add_parser("ingest")
    a.add_argument("--symbols")
    a.add_argument("--period", default=None)
    a.add_argument("--no-news", action="store_true")
    a.set_defaults(func=cmd_ingest)

    a = sub.add_parser("train")
    a.add_argument("--symbols")
    a.add_argument("--horizon", type=int)
    a.add_argument("--model-type", choices=["lightgbm", "xgboost", "gbdt"])
    a.set_defaults(func=cmd_train)

    a = sub.add_parser("predict")
    a.add_argument("symbol", nargs="?")
    a.set_defaults(func=cmd_predict)

    a = sub.add_parser("risk")
    a.add_argument("symbol")
    a.set_defaults(func=cmd_risk)

    a = sub.add_parser("anomalies")
    a.add_argument("symbol")
    a.add_argument("--lookback", type=int, default=180)
    a.set_defaults(func=cmd_anomalies)

    a = sub.add_parser("compare")
    a.add_argument("symbols")
    a.set_defaults(func=cmd_compare)

    a = sub.add_parser("pipeline")
    a.add_argument("--symbols")
    a.add_argument("--period", default=None)
    a.add_argument("--horizon", type=int)
    a.add_argument("--model-type", choices=["lightgbm", "xgboost", "gbdt"])
    a.add_argument("--no-news", action="store_true")
    a.set_defaults(func=cmd_pipeline)

    a = sub.add_parser("serve")
    a.add_argument("--host", default="0.0.0.0")
    a.add_argument("--port", type=int, default=8000)
    a.add_argument("--reload", action="store_true")
    a.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s")
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
