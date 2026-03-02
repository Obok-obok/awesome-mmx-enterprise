from __future__ import annotations

"""Time-based splits for backtesting."""

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def days(self) -> int:
        return int((self.end - self.start).days) + 1


@dataclass(frozen=True)
class HoldoutLastNDaysSplit:
    test_days: int


@dataclass(frozen=True)
class BacktestSplitResult:
    train: DateRange
    test: DateRange


def make_holdout_last_n_days(full: DateRange, split: HoldoutLastNDaysSplit) -> BacktestSplitResult:
    """Split a full range into (train, test) where test is the last N days."""

    n = int(split.test_days)
    if n <= 0:
        raise ValueError("test_days must be > 0")
    if full.days() <= n:
        raise ValueError("Full range too short for requested test_days")

    test_end = full.end
    test_start = full.end - timedelta(days=n - 1)
    train_end = test_start - timedelta(days=1)
    return BacktestSplitResult(train=DateRange(full.start, train_end), test=DateRange(test_start, test_end))
