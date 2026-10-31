"""Train and evaluate one model on one split, then save metrics and predictions.

Usage: uv run python -m psych_crisis.run --model xgboost --split time
"""

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from psych_crisis.config import OUTPUT_DIR, PROJECT_ROOT
from psych_crisis.evaluate import (
    evaluate_predictions,
    fairness_table,
    flag_top_k_per_week,
)
from psych_crisis.events import build_events
from psych_crisis.features import (
    META_COLUMNS,
    build_feature_table,
    feature_columns,
    make_stationary,
)
from psych_crisis.models import baselines
from psych_crisis.splits import patient_folds, time_split

logger = logging.getLogger(__name__)

FitPredict = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]


def _rule_fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    """The rule needs no training."""
    return baselines.rule_score(X_test)


def _fimpp_fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    """Zero-shot FIM-PP: its expected crisis count is the score (needs --extra-features)."""
    return X_test["fimpp_crisis_28d"].fillna(0.0).to_numpy()


def get_model(name: str) -> FitPredict:
    """Look up a model's fit-and-predict function by name (heavy models import lazily)."""
    if name == "rule":
        return _rule_fit_predict
    if name == "fimpp":
        return _fimpp_fit_predict
    if name == "logistic":
        return baselines.logistic_fit_predict
    if name == "xgboost":
        return baselines.xgboost_fit_predict
    if name == "tabpfn":
        from psych_crisis.models.tabpfn_model import tabpfn_fit_predict

        return tabpfn_fit_predict
    if name == "tabpfn35":
        from psych_crisis.models.tabpfn_model import tabpfn35_fit_predict

        return tabpfn35_fit_predict
    if name == "tabpfn_finetuned":
        from psych_crisis.models.tabpfn_model import tabpfn_finetuned_fit_predict

        return tabpfn_finetuned_fit_predict
    raise ValueError(f"Unknown model {name!r}")


def load_feature_table(label: str, extra_features: Path | None = None) -> pd.DataFrame:
    """Load the cached feature table, building it on first use. Optionally join extra score columns."""
    cache = PROJECT_ROOT / "data" / f"features_{label}.parquet"
    if cache.exists():
        table = pd.read_parquet(cache)
    else:
        table = build_feature_table(build_events(), label)
        table.to_parquet(cache)
    if extra_features is not None:
        extra = pd.read_parquet(extra_features)
        table = table.merge(extra, on=["patient_id", "anchor"], how="left", validate="one_to_one")
    return table


def predict_on_split(table: pd.DataFrame, model: FitPredict, split: str) -> pd.DataFrame:
    """Return test-set predictions for the chosen split (pooled over folds for the patient split)."""
    X = baselines.encode(table[feature_columns(table)])
    y = table["target"].to_numpy()
    folds = [time_split(table)] if split == "time" else list(patient_folds(table))
    pieces = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        logger.info("Fold %d: %d train rows, %d test rows", fold, len(train_idx), len(test_idx))
        scores = model(X.iloc[train_idx], y[train_idx], X.iloc[test_idx])
        piece = table.iloc[test_idx][META_COLUMNS + ["age", "male", "setting"]].copy()
        piece["score"] = scores
        piece["fold"] = fold
        pieces.append(piece)
    return pd.concat(pieces).sort_index()


def run(
    model_name: str, split: str, label: str, extra_features: Path | None, tag: str | None, stationary: bool = False
) -> dict:
    """Run one experiment and write outputs/<model>/<split>/."""
    table = load_feature_table(label, extra_features)
    if stationary:
        table = make_stationary(table)
    preds = predict_on_split(table, get_model(model_name), split)
    metrics = evaluate_predictions(preds)
    metrics.update({"model": tag or model_name, "split": split, "label": label})
    if model_name == "rule":
        metrics["brier"] = None  # the rule score is a ratio, not a probability

    out_dir = OUTPUT_DIR / (tag or model_name) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out_dir / "predictions.parquet")
    fairness_table(preds, flag_top_k_per_week(preds, 10)).to_csv(out_dir / "fairness_top10.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(
        "%s / %s: AUROC %.3f %s, PR-AUC %.3f %s (prevalence %.3f)",
        tag or model_name,
        split,
        metrics["auroc"],
        np.round(metrics["ci95"]["auroc"], 3),
        metrics["pr_auc"],
        np.round(metrics["ci95"]["pr_auc"], 3),
        metrics["prevalence"],
    )
    return metrics


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=["time", "patient"], default="time")
    parser.add_argument("--label", choices=["A", "B", "C"], default="A")
    parser.add_argument("--extra-features", type=Path, default=None, help="parquet keyed on patient_id, anchor")
    parser.add_argument("--tag", default=None, help="output folder name (defaults to the model name)")
    parser.add_argument("--stationary", action="store_true", help="replace running totals with yearly rates")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(args.model, args.split, args.label, args.extra_features, args.tag, args.stationary)


if __name__ == "__main__":
    main()
