"""Turn local model outputs into aggregate tables and figures that are safe to publish.

Reads outputs/<model>/<split>/ and data/features_A.parquet (both stay on this machine) and writes
results/*.csv and docs/figures/*.png. Only aggregates are written, and any group with fewer than
11 patients is left out.

Usage: uv run python scripts/make_figures.py
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from psych_crisis.models import baselines
from psych_crisis.splits import time_split

logger = logging.getLogger("make_figures")

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"
MIN_GROUP_PATIENTS = 11

# One colour per model family, used in every figure.
SIMPLE, FOUNDATION, RULE = "#2a78d6", "#eb6834", "#8a8984"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Run folder -> (display name, family). Order is the order shown in the figures.
RUNS = {
    "rule": ("2x own-baseline rule", RULE),
    "logistic": ("Logistic regression", SIMPLE),
    "logistic_stationary": ("Logistic (yearly rates)", SIMPLE),
    "logistic_fimpp": ("Logistic + FIM-PP score", SIMPLE),
    "xgboost": ("XGBoost", SIMPLE),
    "xgboost_stationary": ("XGBoost (yearly rates)", SIMPLE),
    "xgboost_fimpp": ("XGBoost + FIM-PP score", SIMPLE),
    "tabpfn": ("TabPFN-2.5 zero-shot", FOUNDATION),
    "tabpfn_stationary": ("TabPFN-2.5 (yearly rates)", FOUNDATION),
    "tabpfn_finetuned_stationary": ("TabPFN-2.5 fine-tuned", FOUNDATION),
    "tabpfn35_stationary": ("TabPFN-3.5 zero-shot", FOUNDATION),
    "fimpp": ("FIM-PP zero-shot", FOUNDATION),
    "fimpp_finetuned": ("FIM-PP fine-tuned", FOUNDATION),
}
MAIN_RUNS = ("rule", "logistic", "xgboost", "tabpfn", "fimpp")
# Signals compared one at a time (column -> (label, kind)).
SIGNALS = {
    "count_7d": ("Incidents, last 7 days", "raw count"),
    "count_14d": ("Incidents, last 14 days", "raw count"),
    "count_30d": ("Incidents, last 30 days", "raw count"),
    "ratio_14d": ("14-day rate vs own usual rate", "self-relative"),
    "excess_14d": ("14-day count above own usual count", "self-relative"),
    "ratio_30d": ("30-day rate vs own usual rate", "self-relative"),
    "gap_vs_own_median": ("Current gap vs own usual gap", "self-relative"),
    "diversity_30d": ("Different incident types, last 30 days", "raw count"),
    "psych_emergency_30d": ("Psych emergencies, last 30 days", "raw count"),
    "police_30d": ("Police involvement, last 30 days", "raw count"),
    "self_harm_30d": ("Self-harm, last 30 days", "raw count"),
    "aggression_staff_30d": ("Aggression toward staff, last 30 days", "raw count"),
    "days_since_any": ("Days since any incident", "recency"),
    "days_since_psych_emergency": ("Days since last psych emergency", "recency"),
    "days_since_police": ("Days since police involvement", "recency"),
}


def style_axes(ax: plt.Axes) -> None:
    """Recessive axes: no top/right spines, light grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, name: str) -> None:
    """Write one figure at poster-friendly resolution."""
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote %s", name)


def load_metrics() -> pd.DataFrame:
    """One row per (model, split) with the headline numbers."""
    rows = []
    for run, (name, colour) in RUNS.items():
        for split in ("time", "patient"):
            path = OUTPUTS / run / split / "metrics.json"
            if not path.exists():
                continue
            m = json.loads(path.read_text())
            top5, top10 = m["alerts"]["top5_per_week"], m["alerts"]["top10_per_week"]
            rows.append(
                {
                    "run": run,
                    "model": name,
                    "colour": colour,
                    "split": split,
                    "test_rows": m["rows"],
                    "test_patients": m["patients"],
                    "prevalence": m["prevalence"],
                    "auroc": m["auroc"],
                    "auroc_lo": m["ci95"]["auroc"][0],
                    "auroc_hi": m["ci95"]["auroc"][1],
                    "pr_auc": m["pr_auc"],
                    "pr_auc_lo": m["ci95"]["pr_auc"][0],
                    "pr_auc_hi": m["ci95"]["pr_auc"][1],
                    "brier": m["brier"],
                    "top5_crises_caught": top5["crisis_catch_rate"],
                    "top5_false_alarms_per_patient_month": top5[
                        "false_alarms_per_patient_month"
                    ],
                    "top10_crises_caught": top10["crisis_catch_rate"],
                    "top10_false_alarms_per_patient_month": top10[
                        "false_alarms_per_patient_month"
                    ],
                    "top10_precision": top10["precision"],
                }
            )
    return pd.DataFrame(rows)


def write_tables(metrics: pd.DataFrame) -> None:
    """Aggregate results tables for the repo (small groups suppressed)."""
    metrics.drop(columns=["colour"]).round(4).to_csv(
        RESULTS / "summary.csv", index=False
    )
    pieces = []
    for run in ("logistic", "xgboost", "fimpp", "tabpfn", "rule"):
        for split in ("time", "patient"):
            path = OUTPUTS / run / split / "fairness_top10.csv"
            if path.exists():
                table = pd.read_csv(path).assign(model=RUNS[run][0], split=split)
                pieces.append(table[table["patients"] >= MIN_GROUP_PATIENTS])
    pd.concat(pieces).round(4).to_csv(RESULTS / "fairness.csv", index=False)


def plot_auroc_forest(metrics: pd.DataFrame) -> None:
    """Hero figure: AUROC with 95% CI for the main models, one panel per split."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    titles = {
        "time": "Time split (train on the past, test on the last year)",
        "patient": "Patient split (test on unseen people)",
    }
    for ax, split in zip(axes, ("time", "patient")):
        sub = (
            metrics[(metrics["split"] == split) & metrics["run"].isin(MAIN_RUNS)]
            .set_index("run")
            .loc[list(MAIN_RUNS)]
        )
        y = np.arange(len(sub))[::-1]
        ax.hlines(
            y, sub["auroc_lo"], sub["auroc_hi"], colors=sub["colour"], linewidth=2.5
        )
        ax.scatter(
            sub["auroc"],
            y,
            s=70,
            c=sub["colour"],
            edgecolors="white",
            linewidths=1.5,
            zorder=3,
        )
        for yi, (_, row) in zip(y, sub.iterrows()):
            ax.text(
                row["auroc_hi"] + 0.01,
                yi,
                f"{row['auroc']:.2f}",
                va="center",
                fontsize=9,
                color=INK,
            )
        ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
        ax.text(0.505, y[0] + 0.3, "chance", fontsize=8, color=MUTED)
        ax.set_yticks(y, sub["model"])
        ax.set_xlim(0.45, 0.85)
        ax.set_xlabel("AUROC (95% CI)")
        ax.set_title(titles[split], fontsize=10, color=INK, loc="left")
        style_axes(ax)
    fig.suptitle(
        "Every learned model beats chance and the clinician-style rule",
        fontsize=12,
        x=0.01,
        y=1.04,
        ha="left",
    )
    save(fig, "auroc_forest.png")


def plot_alert_tradeoff(metrics: pd.DataFrame) -> None:
    """Crises caught vs false alarms when the team reviews the top 5 or top 10 people each week."""
    sub = metrics[(metrics["split"] == "time") & metrics["run"].isin(MAIN_RUNS)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, k in zip(axes, (5, 10)):
        ax.scatter(
            sub[f"top{k}_false_alarms_per_patient_month"],
            100 * sub[f"top{k}_crises_caught"],
            s=90,
            c=sub["colour"],
            edgecolors="white",
            linewidths=1.5,
            zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["model"],
                (
                    row[f"top{k}_false_alarms_per_patient_month"],
                    100 * row[f"top{k}_crises_caught"],
                ),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
        ax.set_title(
            f"Review the top {k} people each week", fontsize=10, loc="left", color=INK
        )
        ax.set_xlabel("False alarms per person per month")
        ax.margins(x=0.3, y=0.15)
        style_axes(ax)
    axes[0].set_ylabel("Crises flagged beforehand (%)")
    fig.suptitle(
        "A weekly top-10 list flags about half of crises at roughly one false alarm per person every two months",
        fontsize=11,
        x=0.01,
        y=1.03,
        ha="left",
    )
    save(fig, "alerts_tradeoff.png")


def plot_foundation_vs_simple(metrics: pd.DataFrame) -> None:
    """Bigger pretrained models did not beat logistic regression; fine-tuning made them worse."""
    order = [
        "logistic",
        "xgboost",
        "tabpfn",
        "tabpfn_finetuned_stationary",
        "tabpfn35_stationary",
        "fimpp",
        "fimpp_finetuned",
    ]
    sub = metrics[metrics["split"] == "time"].set_index("run").loc[order]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(sub))[::-1]
    ax.barh(y, sub["auroc"] - 0.5, left=0.5, color=sub["colour"], height=0.6)
    for yi, value in zip(y, sub["auroc"]):
        ax.text(value + 0.005, yi, f"{value:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, sub["model"])
    ax.set_xlim(0.5, 0.78)
    ax.set_xlabel("AUROC, time split (0.5 = chance)")
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (SIMPLE, FOUNDATION)]
    ax.legend(
        handles,
        ["Simple models", "Pretrained foundation models"],
        frameon=False,
        loc="lower right",
        fontsize=9,
    )
    ax.set_title(
        "With 125 people, simple models win and fine-tuning overfits",
        fontsize=11,
        loc="left",
    )
    save(fig, "foundation_vs_simple.png")


def plot_pre_crisis_vs_calm(features: pd.DataFrame) -> pd.DataFrame:
    """Same-person comparison: the week right before a crisis vs weeks with no crisis ahead."""
    days_ahead = (features["next_crisis"] - features["anchor"]).dt.days
    before = features[(features["target"] == 1) & (days_ahead <= 7)]
    calm = features[features["target"] == 0]
    rows = []
    for w in (7, 14, 30):
        for group, frame in (("Week before a crisis", before), ("Calm weeks", calm)):
            rows.append(
                {
                    "window_days": w,
                    "group": group,
                    "weeks": len(frame),
                    "patients": frame["patient_id"].nunique(),
                    "any_incident": (frame[f"count_{w}d"] > 0).mean(),
                    "above_2x_own_baseline": (frame[f"ratio_{w}d"] > 2).mean(),
                    "mean_incidents": frame[f"count_{w}d"].mean(),
                }
            )
    table = pd.DataFrame(rows)
    table.round(4).to_csv(RESULTS / "pre_crisis_vs_calm.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, column, title in zip(
        axes,
        ("any_incident", "above_2x_own_baseline"),
        (
            "Any incident in the window",
            "Incidents above 2x the person's own usual rate",
        ),
    ):
        x = np.arange(3)
        for offset, (group, colour) in zip(
            (-0.2, 0.2), (("Week before a crisis", FOUNDATION), ("Calm weeks", SIMPLE))
        ):
            values = 100 * table[table["group"] == group][column].to_numpy()
            ax.bar(x + offset, values, width=0.38, color=colour, label=group)
            for xi, v in zip(x + offset, values):
                ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, ["7 days", "14 days", "30 days"])
        ax.set_title(title, fontsize=10, loc="left")
        style_axes(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of person-weeks (%)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(
        "Behaviour changes before a crisis, but some crises arrive with no warning in the log",
        fontsize=11,
        x=0.01,
        y=1.03,
        ha="left",
    )
    save(fig, "pre_crisis_vs_calm.png")
    return table


def plot_lead_time(features: pd.DataFrame) -> pd.DataFrame:
    """How well the patient-split logistic model ranks crises 1-7, 8-14, 15-21 and 22-28 days away."""
    preds = pd.read_parquet(OUTPUTS / "logistic" / "patient" / "predictions.parquet")
    days_ahead = (preds["next_crisis"] - preds["anchor"]).dt.days
    negatives = preds[preds["target"] == 0]["score"]
    rows = []
    for lo, hi in ((0, 7), (8, 14), (15, 21), (22, 28)):
        pos = preds[(preds["target"] == 1) & days_ahead.between(lo, hi)]
        y = np.r_[np.ones(len(pos)), np.zeros(len(negatives))]
        rows.append(
            {
                "days_until_crisis": f"{max(lo, 1)}-{hi}",
                "crisis_weeks": len(pos),
                "patients": pos["patient_id"].nunique(),
                "auroc_vs_calm_weeks": roc_auc_score(y, np.r_[pos["score"], negatives]),
            }
        )
    table = pd.DataFrame(rows)
    table.round(4).to_csv(RESULTS / "lead_time.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(table))
    ax.plot(
        x,
        table["auroc_vs_calm_weeks"],
        color=SIMPLE,
        linewidth=2,
        marker="o",
        markersize=8,
    )
    for xi, v in zip(x, table["auroc_vs_calm_weeks"]):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, table["days_until_crisis"])
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 0.85)
    ax.set_xlabel("Days until the crisis")
    ax.set_ylabel("AUROC vs calm weeks")
    ax.set_title(
        "The signal is strongest in the last week but is still there 3 to 4 weeks out",
        fontsize=11,
        loc="left",
    )
    style_axes(ax)
    save(fig, "lead_time.png")
    return table


def plot_top_signals(features: pd.DataFrame) -> pd.DataFrame:
    """Single-signal AUROC on the time-split training weeks: how well each signal alone separates crisis weeks."""
    train_idx, _ = time_split(features)
    train = baselines.fill_for_linear(features.iloc[train_idx][list(SIGNALS)])
    y = features["target"].to_numpy()[train_idx]
    rows = []
    for column, (name, kind) in SIGNALS.items():
        auc = roc_auc_score(y, train[column])
        rows.append(
            {
                "feature": column,
                "signal": name,
                "kind": kind,
                "auroc": max(auc, 1 - auc),
                "direction": "higher = more risk"
                if auc >= 0.5
                else "lower = more risk",
            }
        )
    table = pd.DataFrame(rows).sort_values("auroc")
    table.round(4).to_csv(RESULTS / "single_signal_auroc.csv", index=False)

    colours = {"self-relative": FOUNDATION, "raw count": SIMPLE, "recency": RULE}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y_pos = np.arange(len(table))
    ax.barh(
        y_pos,
        table["auroc"] - 0.5,
        left=0.5,
        color=[colours[k] for k in table["kind"]],
        height=0.65,
    )
    for yi, value in zip(y_pos, table["auroc"]):
        ax.text(value + 0.003, yi, f"{value:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y_pos, table["signal"])
    ax.set_xlim(0.5, max(0.75, table["auroc"].max() + 0.03))
    ax.set_xlabel("AUROC of the signal on its own (0.5 = no information)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    ax.legend(
        handles,
        ["Change vs the person's own baseline", "Raw count", "Time since last event"],
        frameon=False,
        fontsize=8,
        loc="lower right",
    )
    ax.set_title(
        "Which signals carry the most warning on their own", fontsize=11, loc="left"
    )
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    save(fig, "top_signals.png")
    return table


def plot_fairness() -> None:
    """True and false positive rates by gender and age band (logistic, time split, top 10 per week)."""
    table = pd.read_csv(OUTPUTS / "logistic" / "time" / "fairness_top10.csv")
    table = table[
        table["attribute"].isin(["gender", "age_band"])
        & (table["patients"] >= MIN_GROUP_PATIENTS)
    ]
    order = {"female": 0, "male": 1, "<25": 2, "25-39": 3, "40+": 4}
    table = table.sort_values("group", key=lambda g: g.map(order))
    labels = [f"{g}\n(n={n})" for g, n in zip(table["group"], table["patients"])]
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(
        x - 0.2,
        100 * table["tpr"],
        width=0.38,
        color=SIMPLE,
        label="Crisis weeks flagged (TPR)",
    )
    ax.bar(
        x + 0.2,
        100 * table["fpr"],
        width=0.38,
        color=FOUNDATION,
        label="Calm weeks flagged (FPR)",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("%")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(
        "Error audit: men's crises are missed more often, women get more false alarms",
        fontsize=11,
        loc="left",
    )
    style_axes(ax)
    ax.grid(axis="x", visible=False)
    save(fig, "fairness.png")


def main() -> None:
    """Build every table and figure."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    plt.rcParams.update(
        {"font.size": 9, "axes.titleweight": "bold", "font.family": "DejaVu Sans"}
    )
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    write_tables(metrics)
    plot_auroc_forest(metrics)
    plot_alert_tradeoff(metrics)
    plot_foundation_vs_simple(metrics)
    features = pd.read_parquet(ROOT / "data" / "features_A.parquet")
    for table in (
        plot_pre_crisis_vs_calm(features),
        plot_lead_time(features),
        plot_top_signals(features),
    ):
        logger.info("\n%s", table.round(3).to_string())
    plot_fairness()


if __name__ == "__main__":
    main()
