"""Bar-setting models: the 2x-own-rate rule, logistic regression and XGBoost."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from psych_crisis.config import RANDOM_SEED

# "Never happened" is filled with a long gap for models that can't take NaN.
NEVER_DAYS = 2000.0


def encode(features: pd.DataFrame) -> pd.DataFrame:
    """One-hot the setting column; everything else is already numeric."""
    encoded = pd.get_dummies(features, columns=["setting"], prefix="setting", dtype=float)
    for setting in ("isl", "group_home", "other_community", "recent_inpatient", "unknown"):
        column = f"setting_{setting}"
        if column not in encoded.columns:
            encoded[column] = 0.0
    return encoded.reindex(sorted(encoded.columns), axis=1)


def rule_score(X_test: pd.DataFrame) -> np.ndarray:
    """Rule baseline: 14-day count relative to the person's own usual rate (flag above 2x)."""
    return X_test["ratio_14d"].to_numpy()


def fill_for_linear(X: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN gaps and log-scale the long-tailed day counts for logistic regression."""
    filled = X.copy()
    day_columns = [c for c in filled.columns if c.startswith("days_since") or c in ("own_median_gap", "history_days")]
    filled[day_columns] = np.log1p(filled[day_columns].fillna(NEVER_DAYS))
    filled["gap_vs_own_median"] = np.log1p(filled["gap_vs_own_median"].fillna(filled["gap_vs_own_median"].median()))
    return filled.fillna(0.0)


def logistic_fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    """L2 logistic regression on scaled features."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    model.fit(fill_for_linear(X_train), y_train)
    return model.predict_proba(fill_for_linear(X_test))[:, 1]


def xgboost_fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    """Small, regularised gradient-boosted trees (NaN handled natively).

    XGBoost is imported here, not at module top: its OpenMP runtime clashes with PyTorch's
    (used by TabPFN) when both load in one process on macOS.
    """
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]
