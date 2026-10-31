"""Build the patient-week prediction table with past-only, self-relative features."""

import logging

import numpy as np
import pandas as pd

from psych_crisis.config import (
    BEHAVIOUR_CATEGORIES,
    HORIZON_DAYS,
    MIN_HISTORY_DAYS,
    WINDOWS,
)
from psych_crisis.labels import mark_crisis_events

logger = logging.getLogger(__name__)

# Per-category counts we keep for the 30-day window.
COUNT_CATEGORIES = (
    "aggression_staff",
    "aggression_peer",
    "threat",
    "property",
    "self_harm",
    "disruptive",
    "elopement",
    "police",
    "psych_emergency",
    "restraint",
    "injury",
)
# "Days since last X" features.
SINCE_CATEGORIES = ("any", "behaviour", "psych_emergency", "police", "restraint", "self_harm")
# Anchors where the person had an inpatient event this recently are skipped (likely in hospital).
RECENT_INPATIENT_DAYS = 30
DAY = np.timedelta64(1, "D")

META_COLUMNS = ["patient_id", "anchor", "target", "label", "next_crisis"]


def make_anchors(events: pd.DataFrame, data_end: pd.Timestamp) -> pd.DataFrame:
    """One anchor per person per Monday, from first record + history buffer to their last usable date."""
    spans = events.groupby("patient_id")["timestamp"].agg(["min", "max"])
    last_usable = data_end - pd.Timedelta(days=HORIZON_DAYS)
    anchors = []
    for patient_id, span in spans.iterrows():
        start = (span["min"] + pd.Timedelta(days=MIN_HISTORY_DAYS)).normalize()
        end = min(span["max"].normalize(), last_usable)
        mondays = pd.date_range(start, end, freq="W-MON")
        anchors.append(pd.DataFrame({"patient_id": patient_id, "anchor": mondays}))
    return pd.concat(anchors, ignore_index=True)


def _count_in_window(times: np.ndarray, anchor: np.datetime64, days: int) -> int:
    """Count sorted timestamps in [anchor - days, anchor)."""
    lo = np.searchsorted(times, anchor - days * DAY, side="left")
    hi = np.searchsorted(times, anchor, side="left")
    return int(hi - lo)


def _days_since(times: np.ndarray, anchor: np.datetime64) -> float:
    """Days since the last timestamp before the anchor, NaN if none."""
    hi = np.searchsorted(times, anchor, side="left")
    if hi == 0:
        return np.nan
    return float((anchor - times[hi - 1]) / DAY)


def _person_timelines(person: pd.DataFrame, crisis: pd.Series) -> dict[str, np.ndarray]:
    """Sorted timestamp arrays for each event type we count, for one person."""
    community = person[~person["inpatient"]]
    stamp = community["timestamp"].to_numpy()
    behaviour = community[list(BEHAVIOUR_CATEGORIES)].any(axis=1).to_numpy()
    timelines = {"any": stamp, "behaviour": stamp[behaviour]}
    for category in COUNT_CATEGORIES:
        timelines[category] = stamp[community[category].to_numpy()]
    timelines["crisis"] = person.loc[crisis.loc[person.index], "timestamp"].to_numpy()
    timelines["inpatient"] = person.loc[person["inpatient"], "timestamp"].to_numpy()
    timelines["category_matrix_times"] = stamp
    return timelines


def _anchor_features(
    timelines: dict[str, np.ndarray],
    categories_matrix: np.ndarray,
    first_seen: np.datetime64,
    anchor: np.datetime64,
) -> dict[str, float]:
    """All past-only features for one person at one anchor."""
    feats: dict[str, float] = {}
    history_days = float((anchor - first_seen) / DAY)
    n_before = np.searchsorted(timelines["any"], anchor, side="left")
    own_daily_rate = n_before / max(history_days, 1.0)
    feats["history_days"] = history_days
    feats["n_events_total"] = float(n_before)
    feats["n_crisis_total"] = float(np.searchsorted(timelines["psych_emergency"], anchor, side="left"))

    for days in WINDOWS:
        count = _count_in_window(timelines["any"], anchor, days)
        expected = own_daily_rate * days
        feats[f"count_{days}d"] = count
        feats[f"behaviour_{days}d"] = _count_in_window(timelines["behaviour"], anchor, days)
        feats[f"ratio_{days}d"] = (count + 0.5) / (expected + 0.5)
        feats[f"excess_{days}d"] = count - expected

    for category in COUNT_CATEGORIES:
        feats[f"{category}_30d"] = _count_in_window(timelines[category], anchor, 30)

    for category in SINCE_CATEGORIES:
        source = timelines["any"] if category == "any" else timelines[category]
        feats[f"days_since_{category}"] = _days_since(source, anchor)

    past = timelines["any"][:n_before]
    gaps = np.diff(past) / DAY if len(past) > 1 else np.array([])
    feats["own_median_gap"] = float(np.median(gaps)) if len(gaps) else np.nan
    since_any = feats["days_since_any"]
    feats["gap_vs_own_median"] = (
        since_any / feats["own_median_gap"] if len(gaps) and feats["own_median_gap"] > 0 else np.nan
    )

    lo = np.searchsorted(timelines["category_matrix_times"], anchor - 30 * DAY, side="left")
    hi = np.searchsorted(timelines["category_matrix_times"], anchor, side="left")
    feats["diversity_30d"] = float(categories_matrix[lo:hi].any(axis=0).sum())
    feats["days_since_inpatient"] = _days_since(timelines["inpatient"], anchor)
    return feats


def _target_and_eligibility(
    crisis_times: np.ndarray, anchor: np.datetime64
) -> tuple[int, bool, np.datetime64 | None]:
    """Target = any crisis in [anchor, anchor + horizon). Eligible only if no crisis in the prior 7 days.

    Also returns the date of the crisis that makes the target positive (None if negative).
    """
    in_prior_week = _count_in_window(crisis_times, anchor, 7) > 0
    lo = np.searchsorted(crisis_times, anchor, side="left")
    hi = np.searchsorted(crisis_times, anchor + HORIZON_DAYS * DAY, side="left")
    next_crisis = crisis_times[lo] if hi > lo else None
    return int(hi > lo), not in_prior_week, next_crisis


def _static_features(person: pd.DataFrame, anchor: pd.Timestamp) -> dict[str, float | str]:
    """Age, gender and most recent setting known at the anchor."""
    past = person[person["timestamp"] < anchor]
    latest = past.iloc[-1]
    years_elapsed = (anchor - latest["timestamp"]).days / 365.25
    return {
        "age": float(latest["age"]) + years_elapsed,
        "male": float(latest["gender"] == "Male"),
        "setting": latest["setting"] if latest["setting"] != "inpatient" else "recent_inpatient",
    }


def build_feature_table(events: pd.DataFrame, label: str = "A") -> pd.DataFrame:
    """Full step 2: one row per eligible patient-week with features and 28-day target."""
    data_end = events["timestamp"].max()
    crisis = mark_crisis_events(events, label)
    anchors = make_anchors(events, data_end)
    category_columns = list(BEHAVIOUR_CATEGORIES) + ["police", "psych_emergency", "medical"]

    rows = []
    skipped_inpatient = 0
    for patient_id, person in events.groupby("patient_id", sort=False):
        timelines = _person_timelines(person, crisis)
        community = person[~person["inpatient"]]
        categories_matrix = community[category_columns].to_numpy()
        first_seen = person["timestamp"].min().to_datetime64()
        for anchor in anchors.loc[anchors["patient_id"] == patient_id, "anchor"]:
            anchor64 = anchor.to_datetime64()
            if _days_since(timelines["inpatient"], anchor64) <= RECENT_INPATIENT_DAYS:
                skipped_inpatient += 1
                continue
            target, eligible, next_crisis = _target_and_eligibility(timelines["crisis"], anchor64)
            if not eligible:
                continue
            row = {
                "patient_id": patient_id,
                "anchor": anchor,
                "target": target,
                "label": label,
                "next_crisis": pd.Timestamp(next_crisis) if next_crisis is not None else pd.NaT,
            }
            row.update(_static_features(person, anchor))
            row.update(_anchor_features(timelines, categories_matrix, first_seen, anchor64))
            rows.append(row)

    table = pd.DataFrame(rows)
    logger.info(
        "Feature table: %d rows, %d patients, %.1f%% positive (%d anchors skipped as recent inpatient)",
        len(table),
        table["patient_id"].nunique(),
        100 * table["target"].mean(),
        skipped_inpatient,
    )
    return table


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Model input columns (everything except identifiers and target)."""
    return [c for c in table.columns if c not in META_COLUMNS]


# Columns that keep growing with calendar time, so a time-split test set sits outside the training range.
CUMULATIVE_COLUMNS = ("history_days", "n_events_total", "n_crisis_total")


def make_stationary(table: pd.DataFrame) -> pd.DataFrame:
    """Swap running totals for yearly rates so features don't drift with calendar time."""
    years = table["history_days"].clip(lower=1.0) / 365.25
    stationary = table.assign(
        events_per_year=table["n_events_total"] / years,
        crises_per_year=table["n_crisis_total"] / years,
    )
    return stationary.drop(columns=list(CUMULATIVE_COLUMNS))
