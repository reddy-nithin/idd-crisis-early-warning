"""Export event timelines and anchors for FIM-PP, which runs in its own environment.

FIM-PP needs one mark per event, so each event gets its most serious category (priority order below).
Usage: uv run python -m psych_crisis.fimpp_export
"""

import logging
import pickle

import numpy as np
import pandas as pd

from psych_crisis.config import HORIZON_DAYS, PROJECT_ROOT, RANDOM_SEED, TIME_SPLIT_DATE
from psych_crisis.events import build_events
from psych_crisis.run import load_feature_table
from psych_crisis.splits import patient_folds

logger = logging.getLogger(__name__)

EXPORT_DIR = PROJECT_ROOT / "data" / "fimpp"
# Mark id = position in this list; the first matching category wins.
MARK_PRIORITY: list[tuple[str, tuple[str, ...]]] = [
    ("psych_emergency", ("psych_emergency",)),
    ("police", ("police",)),
    ("self_harm", ("self_harm",)),
    ("aggression", ("aggression_staff", "aggression_peer", "threat", "property")),
    ("other", ()),
]


def assign_mark(event: pd.Series) -> int:
    """Index of the first mark whose categories appear in the event (falls back to 'other')."""
    for mark_id, (_, categories) in enumerate(MARK_PRIORITY):
        if any(event[c] for c in categories):
            return mark_id
    return len(MARK_PRIORITY) - 1


def export_sequences(events: pd.DataFrame) -> pd.DataFrame:
    """Community events only, one mark each."""
    community = events[~events["inpatient"]].copy()
    community["mark"] = community.apply(assign_mark, axis=1)
    return community[["patient_id", "timestamp", "mark"]].sort_values(["patient_id", "timestamp"])


def export_anchors(table: pd.DataFrame) -> pd.DataFrame:
    """Anchors with their patient-split fold, so FIM-PP context can exclude the scored fold."""
    anchors = table[["patient_id", "anchor"]].copy()
    anchors["fold"] = -1
    for fold, (_, test_idx) in enumerate(patient_folds(table)):
        anchors.iloc[test_idx, anchors.columns.get_loc("fold")] = fold
    return anchors


FINETUNE_DIR = EXPORT_DIR / "finetune_time"
DEV_FRACTION = 0.2
MAX_EVENTS = 100


def _to_cdiff(person: pd.DataFrame) -> list[dict]:
    """One timeline in OpenFIM's CDiff format (days since that person's first kept event)."""
    person = person.tail(MAX_EVENTS)
    days = (person["timestamp"] - person["timestamp"].iloc[0]) / pd.Timedelta(days=1)
    return [{"time_since_start": float(t), "type_event": int(m)} for t, m in zip(days, person["mark"])]


def export_finetune_data(sequences: pd.DataFrame) -> dict[str, int]:
    """train.pkl / dev.pkl for FIM-PP fine-tuning, using only events before the time-split purge date.

    Dev holds out 20% of people entirely, so validation never sees a training person's timeline.
    """
    cutoff = pd.Timestamp(TIME_SPLIT_DATE) - pd.Timedelta(days=HORIZON_DAYS)
    early = sequences[sequences["timestamp"] < cutoff]
    timelines = {pid: _to_cdiff(grp) for pid, grp in early.groupby("patient_id") if len(grp) >= 2}
    patients = np.array(sorted(timelines))
    rng = np.random.default_rng(RANDOM_SEED)
    dev_patients = set(rng.choice(patients, size=int(len(patients) * DEV_FRACTION), replace=False))
    splits = {
        "train": [timelines[p] for p in patients if p not in dev_patients],
        "dev": [timelines[p] for p in patients if p in dev_patients],
    }
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    for name, seqs in splits.items():
        with open(FINETUNE_DIR / f"{name}.pkl", "wb") as handle:
            pickle.dump({name: seqs, "dim_process": len(MARK_PRIORITY)}, handle)
    return {name: len(seqs) for name, seqs in splits.items()}


def main() -> None:
    """Write sequences.parquet and anchors.parquet under data/fimpp/."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    sequences = export_sequences(build_events())
    anchors = export_anchors(load_feature_table("A"))
    sequences.to_parquet(EXPORT_DIR / "sequences.parquet")
    anchors.to_parquet(EXPORT_DIR / "anchors.parquet")
    logger.info("Exported %d events (marks: %s) and %d anchors", len(sequences),
                sequences["mark"].value_counts().sort_index().to_dict(), len(anchors))
    logger.info("Fine-tuning timelines (before %s minus %d days): %s",
                TIME_SPLIT_DATE, HORIZON_DAYS, export_finetune_data(sequences))


if __name__ == "__main__":
    main()
