"""Load the incident CSV and collapse it to one row per event."""

import logging
from pathlib import Path

import pandas as pd

from psych_crisis.config import (
    EVENT_FLAGS,
    INCIDENT_CATEGORIES,
    INPATIENT_PREFIX,
    RAW_CSV,
)

logger = logging.getLogger(__name__)

EVENT_KEY = ["patient_id", "timestamp"]


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Read the raw export (one row per incident type within an event)."""
    if not path.exists():
        raise FileNotFoundError(f"Incident CSV not found at {path}")
    raw = pd.read_csv(path, encoding="cp1252", na_values="NULL")
    logger.info("Loaded %d raw rows from %s", len(raw), path.name)
    return raw


def map_incident_category(incident_type: str) -> str:
    """Return the category for a raw incident type, or 'other' if unknown."""
    for category, types in INCIDENT_CATEGORIES.items():
        if incident_type in types:
            return category
    logger.warning("Unmapped incident type %r, using 'other'", incident_type)
    return "other"


def tidy_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate reports and add timestamp, category and flag columns per row."""
    rows = raw[raw["IsDuplicateEvent"] != "Y"].copy()
    rows["patient_id"] = rows["StudyPatientID"]
    date = pd.to_datetime(rows["EventDate"]).dt.normalize()
    time = pd.to_timedelta(rows["EventTime"].fillna("0:00:00"))
    rows["timestamp"] = date + time
    rows["category"] = rows["IncidentType"].map(map_incident_category)
    rows["restraint"] = rows["IsEP-Event"].eq("Y") | rows["EP Type"].notna()
    rows["injury"] = rows["IsInjury-ConsumerModule"].eq("Y")
    rows["er_required"] = rows["IsEmergencyRoomRequired-FromInjuryModule"].eq("Y")
    rows["hospitalized"] = rows["InjurySeverity"].eq("Hospitalization")
    rows["staff_injury"] = rows["IsStaffInjury"].eq("Y")
    rows["inpatient"] = rows["ProgramCategory"].fillna("").str.startswith(INPATIENT_PREFIX)
    return rows


def collapse_to_events(rows: pd.DataFrame) -> pd.DataFrame:
    """Turn tidy rows into one row per (patient, timestamp) with multi-hot category flags."""
    category_flags = pd.crosstab(
        [rows["patient_id"], rows["timestamp"]], rows["category"]
    ).clip(upper=1)
    for category in INCIDENT_CATEGORIES:
        if category not in category_flags.columns:
            category_flags[category] = 0
    category_flags = category_flags[list(INCIDENT_CATEGORIES)].astype(bool)

    grouped = rows.groupby(EVENT_KEY)
    flags = grouped[list(EVENT_FLAGS) + ["inpatient"]].any()
    info = grouped.agg(
        gender=("ConsumerGender", "first"),
        age=("AgeasofEvent", "first"),
        program=("ProgramCategory", "first"),
        residential=("Residential Consumer", "first"),
        n_incident_types=("IncidentType", "nunique"),
    )
    events = info.join(flags).join(category_flags).reset_index()
    events["setting"] = events["program"].map(program_to_setting)
    events = events.sort_values(EVENT_KEY).reset_index(drop=True)
    logger.info("Collapsed %d rows into %d events", len(rows), len(events))
    return events


def program_to_setting(program: str | float) -> str:
    """Reduce the program name to a coarse setting type (no organisation names)."""
    if not isinstance(program, str):
        return "unknown"
    if program.startswith(INPATIENT_PREFIX):
        return "inpatient"
    if "ISL" in program:
        return "isl"
    if "Group Home" in program:
        return "group_home"
    return "other_community"


def build_events(path: Path = RAW_CSV) -> pd.DataFrame:
    """Full step 1: raw CSV to the event table."""
    return collapse_to_events(tidy_rows(load_raw(path)))
