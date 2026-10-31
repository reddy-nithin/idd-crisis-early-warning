"""Metrics: discrimination with patient-level bootstrap CIs, alert workload, and fairness."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

N_BOOTSTRAP = 1000
# How many people the team could review each week (alert budget).
WEEKLY_ALERT_BUDGETS = (5, 10)
WEEKS_PER_MONTH = 4.345


def _point_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """AUROC, PR-AUC and Brier for one sample."""
    return {
        "auroc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, np.clip(score, 0, 1))),
    }


def bootstrap_ci(preds: pd.DataFrame, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> dict[str, list[float]]:
    """95% CIs by resampling patients (not rows), since rows of one person are correlated."""
    rng = np.random.default_rng(seed)
    by_patient = {pid: grp for pid, grp in preds.groupby("patient_id")}
    patients = np.array(list(by_patient))
    samples: dict[str, list[float]] = {"auroc": [], "pr_auc": [], "brier": []}
    for _ in range(n_boot):
        chosen = rng.choice(patients, size=len(patients), replace=True)
        sample = pd.concat([by_patient[p] for p in chosen])
        if sample["target"].nunique() < 2:
            continue
        for name, value in _point_metrics(sample["target"].to_numpy(), sample["score"].to_numpy()).items():
            samples[name].append(value)
    return {name: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for name, v in samples.items()}


def flag_top_k_per_week(preds: pd.DataFrame, k: int) -> pd.Series:
    """Flag the k highest-scoring people in each anchor week."""
    rank = preds.groupby("anchor")["score"].rank(method="first", ascending=False)
    return rank <= k


def alert_summary(preds: pd.DataFrame, flagged: pd.Series) -> dict[str, float]:
    """Crises caught (any flag in the 28 days before) and false alarms per patient-month."""
    positives = preds[preds["target"] == 1]
    crises = positives.groupby(["patient_id", "next_crisis"]).apply(
        lambda g: bool(flagged.loc[g.index].any()), include_groups=False
    )
    negatives = preds["target"] == 0
    patient_months = len(preds) / WEEKS_PER_MONTH
    return {
        "crises_total": len(crises),
        "crises_caught": int(crises.sum()),
        "crisis_catch_rate": float(crises.mean()) if len(crises) else float("nan"),
        "week_sensitivity": float(flagged[preds["target"] == 1].mean()),
        "false_alarms": int((flagged & negatives).sum()),
        "false_alarms_per_patient_month": float((flagged & negatives).sum() / patient_months),
        "precision": float(preds.loc[flagged, "target"].mean()) if flagged.any() else float("nan"),
    }


def age_band(age: pd.Series) -> pd.Series:
    """Coarse age groups for the fairness table."""
    return pd.cut(age, bins=[0, 25, 40, 120], labels=["<25", "25-39", "40+"], right=False).astype(str)


def fairness_table(preds: pd.DataFrame, flagged: pd.Series) -> pd.DataFrame:
    """True- and false-positive rates by gender, age band and setting (CAMH-style equalized odds check)."""
    groups = {
        "gender": preds["male"].map({1.0: "male", 0.0: "female"}),
        "age_band": age_band(preds["age"]),
        "setting": preds["setting"],
    }
    rows = []
    for attribute, values in groups.items():
        for value in sorted(values.unique()):
            mask = values == value
            y, f = preds.loc[mask, "target"], flagged[mask]
            rows.append(
                {
                    "attribute": attribute,
                    "group": value,
                    "patients": int(preds.loc[mask, "patient_id"].nunique()),
                    "weeks": int(mask.sum()),
                    "positive_weeks": int(y.sum()),
                    "tpr": float(f[y == 1].mean()) if (y == 1).any() else float("nan"),
                    "fpr": float(f[y == 0].mean()) if (y == 0).any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def evaluate_predictions(preds: pd.DataFrame, n_boot: int = N_BOOTSTRAP) -> dict:
    """All metrics for one model's test predictions."""
    y, score = preds["target"].to_numpy(), preds["score"].to_numpy()
    result: dict = {
        "rows": len(preds),
        "patients": int(preds["patient_id"].nunique()),
        "prevalence": float(y.mean()),
        **_point_metrics(y, score),
        "ci95": bootstrap_ci(preds, n_boot),
        "alerts": {},
    }
    for k in WEEKLY_ALERT_BUDGETS:
        result["alerts"][f"top{k}_per_week"] = alert_summary(preds, flag_top_k_per_week(preds, k))
    return result
