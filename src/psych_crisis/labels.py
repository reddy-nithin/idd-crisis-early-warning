"""Crisis definitions and crisis-onset detection."""

import pandas as pd

from psych_crisis.config import MIN_GAP_DAYS

# Option A: psychiatric emergency only. B adds police. C adds ER or hospitalisation.
LABEL_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "A": ("psych_emergency",),
    "B": ("psych_emergency", "police"),
    "C": ("psych_emergency", "police", "er_required", "hospitalized"),
}


def mark_crisis_events(events: pd.DataFrame, label: str = "A") -> pd.Series:
    """Return a boolean Series: does each community event meet the crisis definition?"""
    if label not in LABEL_DEFINITIONS:
        raise ValueError(f"Unknown label {label!r}, expected one of {list(LABEL_DEFINITIONS)}")
    is_crisis = events[list(LABEL_DEFINITIONS[label])].any(axis=1)
    return is_crisis & ~events["inpatient"]


def find_onsets(events: pd.DataFrame, label: str = "A", min_gap_days: int = MIN_GAP_DAYS) -> pd.DataFrame:
    """Return crisis onsets: crisis events at least min_gap_days after that person's previous crisis.

    A person's first crisis is always an onset.
    """
    crises = events[mark_crisis_events(events, label)][["patient_id", "timestamp"]].copy()
    crises = crises.sort_values(["patient_id", "timestamp"])
    gap = crises.groupby("patient_id")["timestamp"].diff()
    is_onset = gap.isna() | (gap >= pd.Timedelta(days=min_gap_days))
    onsets = crises[is_onset].reset_index(drop=True)
    return onsets
