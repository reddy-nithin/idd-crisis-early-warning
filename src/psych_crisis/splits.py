"""Train/test splits: time-ordered, and grouped by patient."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from psych_crisis.config import HORIZON_DAYS, TIME_SPLIT_DATE


def time_split(table: pd.DataFrame, cut: str = TIME_SPLIT_DATE) -> tuple[np.ndarray, np.ndarray]:
    """Train on anchors whose 28-day window ends before the cut, test on anchors from the cut on.

    The purge gap stops training targets from overlapping the test period.
    """
    cut_date = pd.Timestamp(cut)
    train = table["anchor"] < cut_date - pd.Timedelta(days=HORIZON_DAYS)
    test = table["anchor"] >= cut_date
    return np.flatnonzero(train), np.flatnonzero(test)


def patient_folds(table: pd.DataFrame, n_folds: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Grouped K-fold: every patient is in exactly one test fold."""
    splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=0)
    yield from splitter.split(table, table["target"], groups=table["patient_id"])
