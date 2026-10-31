"""Paths and study settings shared by the whole pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "incidents.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Look-back windows (days) used for counts and personal baselines.
WINDOWS = (7, 14, 30)
# Predict a crisis onset within this many days of the anchor date.
HORIZON_DAYS = 28
# A psychiatric emergency counts as a new onset only after this many quiet days.
MIN_GAP_DAYS = 7
# Each person needs this much history before we start predicting for them.
MIN_HISTORY_DAYS = 90
# Train on anchors before this date, test on anchors from it onward.
TIME_SPLIT_DATE = "2025-10-01"
RANDOM_SEED = 42

INPATIENT_PREFIX = "Adult Inpatient"

# 48 raw incident types grouped into categories. These double as FIM-PP marks (max 22).
INCIDENT_CATEGORIES: dict[str, list[str]] = {
    "psych_emergency": ["Psychiatric Emergency - DD Consumer"],
    "police": ["Law Enforcement Involvement"],
    "aggression_staff": [
        "Physical Altercation - Staff & Consumer",
        "Assault without Serious Injury (IRA)",
        "Assault with Serious Injury (IRA)",
    ],
    "aggression_peer": [
        "Physical Altercation between Consumers",
        "Physical Altercation - Consumer & Non-Staff",
    ],
    "threat": ["Graphic Threat of Harm", "Graphic Threat of Physical Violence (IRA)"],
    "property": [
        "Consumer Struck Object",
        "Property Loss/Destruction",
        "Significant Property Destruction (IRA)",
        "Theft",
    ],
    "self_harm": [
        "Consumer Self Harm",
        "Self-Injurious Behavior (IRA)",
        "Suicide Attempt",
        "Ingestion of non-Food Item",
    ],
    "restraint_report": ["Emergency Procedures", "Prohibited Procedure Used"],
    "disruptive": ["Disruptive Behavior"],
    "elopement": ["Elopement", "Elopement-Unlocked", "Unauthorized Absence (not elopement)"],
    "sexual": ["Sexual Conduct-Consumer-non-Consensual", "Sexual Activity Where Prohibited"],
    "medical": [
        "Medical Emergency - Consumer",
        "Seizure(s)",
        "Fall (witnessed)",
        "Fall (unwitnessed)",
        "Accident",
        "Choking",
        "Infection Control Issue",
        "Injury to Consumer - Unknown Origin",
        "Vehicular Accident",
        "Death",
    ],
    "substance": ["Substance Abuse (IRA)", "Possession of Contraband"],
    "staff_or_system": [
        "Employee Misconduct Allegation",
        "Sexual Conduct between Staff & Consumer",
        "Inappropriate Language by Staff to Consumer",
        "Medication Error",
        "Supervision Level Not Followed",
        "Consumer Rights",
        "Misuse of Consumer's Funds/Property",
        "DMH Staff Injury",
        "Medical Emergency - DMH Staff",
        "Safety or Security Issue",
    ],
    "other": ["Other"],
}

# Categories that describe the person's own behaviour (used for "behaviour" counts).
BEHAVIOUR_CATEGORIES = (
    "aggression_staff",
    "aggression_peer",
    "threat",
    "property",
    "self_harm",
    "disruptive",
    "elopement",
    "sexual",
    "substance",
)

# Extra event-level flags built from the injury and restraint columns.
EVENT_FLAGS = ("restraint", "injury", "er_required", "hospitalized", "staff_injury")
