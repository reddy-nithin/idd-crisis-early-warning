"""Remove staff free text from the working copy of the incident CSV.

Two columns carry typed notes: "Other (note)" in IncidentTypeList and "Other: note" in
InjuredBodyPartList. Both are cut back to plain "Other". Every other cell is left byte-for-byte as is.
The original export is kept, untouched, outside the repo.

Usage: uv run python scripts/strip_free_text.py
"""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger("strip_free_text")

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "incidents.csv"
FREE_TEXT_COLUMNS = ("IncidentTypeList", "InjuredBodyPartList")
# "Other (note)" -> "Other", "Other: note (Other: note)" -> "Other (Other)", "(Other: note)" -> "(Other)"
INCIDENT_NOTE = re.compile(r"Other \([^)]*\)")
INJURY_NOTE = re.compile(r"Other: [^;()]*")


def strip_cell(value: str) -> str:
    """Drop every typed note from one cell, keeping the fixed vocabulary around it."""
    return INJURY_NOTE.sub("Other", INCIDENT_NOTE.sub("Other", value))


def main() -> None:
    """Rewrite the working CSV in place, keeping its encoding and NULL markers."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw = pd.read_csv(CSV_PATH, encoding="cp1252", dtype=str, keep_default_na=False)
    for column in FREE_TEXT_COLUMNS:
        before = raw[column].copy()
        raw[column] = raw[column].map(strip_cell)
        logger.info("%s: %d cells changed", column, int((before != raw[column]).sum()))
    leftover = raw[list(FREE_TEXT_COLUMNS)].apply(lambda col: col.str.contains(r"Other:|Other \((?!Other\))", regex=True))
    if leftover.to_numpy().any():
        raise ValueError("Free text still present after stripping")
    raw.to_csv(CSV_PATH, encoding="cp1252", index=False)
    logger.info("Wrote %s", CSV_PATH)


if __name__ == "__main__":
    main()
