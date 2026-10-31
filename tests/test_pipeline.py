"""Tests for event collapsing, crisis labels and past-only features on a hand-built timeline."""

import numpy as np
import pandas as pd
import pytest

from psych_crisis.events import collapse_to_events, tidy_rows
from psych_crisis.features import build_feature_table, feature_columns
from psych_crisis.labels import find_onsets

PE = "Psychiatric Emergency - DD Consumer"
HIT = "Physical Altercation - Staff & Consumer"
POLICE = "Law Enforcement Involvement"


def raw_row(patient: str, date: str, time: str, incident: str, program: str = "ISL", dup: str = "N") -> dict:
    """One raw CSV row with only the columns the pipeline reads."""
    return {
        "StudyPatientID": patient,
        "EventDate": f"{date} 0:00:00",
        "EventTime": time,
        "IncidentType": incident,
        "IsDuplicateEvent": dup,
        "ProgramCategory": program,
        "Residential Consumer": "Y",
        "IsEP-Event": "N",
        "EP Type": np.nan,
        "IsInjury-ConsumerModule": "N",
        "IsEmergencyRoomRequired-FromInjuryModule": np.nan,
        "InjurySeverity": np.nan,
        "IsStaffInjury": "N",
        "ConsumerGender": "Male",
        "AgeasofEvent": 30,
    }


def build(rows: list[dict]) -> pd.DataFrame:
    """Raw rows to the event table."""
    return collapse_to_events(tidy_rows(pd.DataFrame(rows)))


@pytest.fixture
def timeline() -> list[dict]:
    """One person: early history, a build-up of incidents, then a crisis."""
    return [
        raw_row("P1", "2024-01-01", "10:00:00", HIT),
        raw_row("P1", "2024-02-01", "10:00:00", HIT),
        raw_row("P1", "2024-05-20", "09:00:00", HIT),
        raw_row("P1", "2024-05-20", "09:00:00", POLICE),
        raw_row("P1", "2024-05-25", "09:00:00", HIT),
        raw_row("P1", "2024-06-05", "12:00:00", PE),
        raw_row("P1", "2024-06-08", "12:00:00", PE),
        raw_row("P1", "2024-08-01", "12:00:00", PE),
        raw_row("P1", "2024-09-30", "12:00:00", HIT),
    ]


def test_collapse_merges_rows_of_one_event(timeline: list[dict]) -> None:
    events = build(timeline)
    assert len(events) == 8
    merged = events[events["timestamp"] == pd.Timestamp("2024-05-20 09:00")].iloc[0]
    assert merged["aggression_staff"] and merged["police"]


def test_duplicate_reports_are_dropped(timeline: list[dict]) -> None:
    rows = timeline + [raw_row("P1", "2024-03-01", "10:00:00", HIT, dup="Y")]
    assert len(build(rows)) == 8


def test_onsets_need_seven_quiet_days(timeline: list[dict]) -> None:
    onsets = find_onsets(build(timeline), "A")
    assert list(onsets["timestamp"].dt.strftime("%Y-%m-%d")) == ["2024-06-05", "2024-08-01"]


def test_inpatient_crisis_is_not_a_label(timeline: list[dict]) -> None:
    rows = timeline + [raw_row("P1", "2024-07-01", "12:00:00", PE, program="Adult Inpatient - Example Unit")]
    onsets = find_onsets(build(rows), "A")
    assert pd.Timestamp("2024-07-01 12:00") not in set(onsets["timestamp"])


def test_label_b_adds_police(timeline: list[dict]) -> None:
    onsets = find_onsets(build(timeline), "B")
    assert pd.Timestamp("2024-05-20 09:00") in set(onsets["timestamp"])


def test_target_and_features_at_known_anchor(timeline: list[dict]) -> None:
    table = build_feature_table(build(timeline), "A").set_index("anchor")
    row = table.loc[pd.Timestamp("2024-06-03")]
    assert row["target"] == 1  # crisis on Jun 5, inside 28 days
    assert row["count_14d"] == 2  # May 20 and May 25
    assert row["police_30d"] == 1
    assert row["days_since_psych_emergency"] != row["days_since_psych_emergency"]  # NaN: no past crisis


def test_anchor_right_after_crisis_is_excluded(timeline: list[dict]) -> None:
    table = build_feature_table(build(timeline), "A")
    assert pd.Timestamp("2024-06-10") not in set(table["anchor"])  # crisis on Jun 8 = prior week


def test_features_ignore_the_future(timeline: list[dict]) -> None:
    base = build_feature_table(build(timeline), "A").set_index("anchor")
    future_rows = timeline + [raw_row("P1", "2024-06-04", "08:00:00", HIT)] * 1
    future = build_feature_table(build(future_rows), "A").set_index("anchor")
    anchor = pd.Timestamp("2024-06-03")
    cols = [c for c in feature_columns(base.reset_index()) if c not in ("anchor",)]
    pd.testing.assert_series_equal(base.loc[anchor, cols], future.loc[anchor, cols], check_names=False)


def test_fimpp_mark_priority(timeline: list[dict]) -> None:
    from psych_crisis.fimpp_export import export_sequences

    marks = export_sequences(build(timeline)).set_index("timestamp")["mark"]
    assert marks[pd.Timestamp("2024-05-20 09:00")] == 1  # police outranks aggression
    assert marks[pd.Timestamp("2024-06-05 12:00")] == 0  # crisis
    assert marks[pd.Timestamp("2024-01-01 10:00")] == 3  # aggression


def test_tabpfn_context_keeps_every_positive() -> None:
    from psych_crisis.models.tabpfn_model import subsample_context

    y = np.array([1] * 30 + [0] * 970)
    X = pd.DataFrame({"a": np.arange(1000, dtype=float)})
    X_ctx, y_ctx = subsample_context(X, y, max_rows=100)
    assert len(y_ctx) == 100 and y_ctx.sum() == 30
