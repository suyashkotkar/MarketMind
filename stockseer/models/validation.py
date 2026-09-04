"""Time-aware validation.

A plain KFold on financial panel data leaks in two ways:
  1. future rows train the model that scores past rows;
  2. the h-day forward target of the last train rows overlaps the first test rows.

`PurgedWalkForward` fixes both: folds always move forward in time, and an
`embargo` of at least the prediction horizon is cut out between train and test.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PurgedWalkForward:
    n_splits: int = 5
    embargo: int = 5
    min_train_size: int = 250
    expanding: bool = True

    def split(self, dates: pd.Series) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) as positional indices into `dates`.

        `dates` may repeat (a panel has one row per symbol per day); splits are
        made on unique dates so a day is never partly in train and partly in test.
        """
        d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        uniq = np.array(sorted(d.unique()))
        n = len(uniq)
        if n < self.n_splits + 2:
            raise ValueError(f"need >= {self.n_splits + 2} distinct dates, got {n}")

        fold = n // (self.n_splits + 1)
        for k in range(1, self.n_splits + 1):
            train_end = fold * k
            test_start = train_end + self.embargo
            test_end = min(fold * (k + 1) + (n % (self.n_splits + 1)
                                             if k == self.n_splits else 0), n)
            if test_start >= test_end:
                continue
            train_start = 0 if self.expanding else max(
                0, train_end - self.min_train_size)

            train_dates = set(uniq[train_start:train_end])
            test_dates = set(uniq[test_start:test_end])
            tr = np.flatnonzero(d.isin(train_dates).values)
            te = np.flatnonzero(d.isin(test_dates).values)
            if len(tr) >= self.min_train_size and len(te) > 0:
                yield tr, te

    def get_n_splits(self, *_args, **_kwargs) -> int:
        return self.n_splits


def train_test_split_by_date(df: pd.DataFrame, test_fraction: float = 0.2,
                             embargo: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A single chronological holdout with an embargo gap."""
    d = pd.to_datetime(df["date"])
    uniq = np.array(sorted(d.unique()))
    cut = int(len(uniq) * (1 - test_fraction))
    train_dates = set(uniq[:cut])
    test_dates = set(uniq[min(cut + embargo, len(uniq)):])
    return df[d.isin(train_dates)].copy(), df[d.isin(test_dates)].copy()
