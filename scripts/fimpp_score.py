"""Zero-shot FIM-PP crisis scores for every anchor. Runs in the OpenFIM environment, not ours.

For each person, the context is other people's timelines (never their own), and the inference
path is that person's events before the anchor. The score is the expected number of psychiatric
emergencies in the next 28 days (integral of the mark-0 intensity).

Usage:
  ~/projects/vendor/OpenFIM/.venv/bin/python -I scripts/fimpp_score.py --split time
  ~/projects/vendor/OpenFIM/.venv/bin/python -I scripts/fimpp_score.py --split patient
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

logger = logging.getLogger("fimpp_score")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = PROJECT_ROOT / "data" / "fimpp"
TIME_SPLIT_DATE = pd.Timestamp("2025-10-01")
MAX_EVENTS = 100  # FIM-PP was pretrained on sequences of 15 to 100 events
HORIZON_DAYS = 28
PAD_TIME = 1e6  # padding must sort after real events (the model binary-searches event times)
NUM_MARKS = 5
CRISIS_MARK = 0


def to_days(stamps: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    """Timestamps as float days since an origin."""
    return ((stamps - origin) / pd.Timedelta(days=1)).to_numpy(dtype=np.float32)


def pad_paths(paths: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack (times, marks) paths into padded [1, P, L, 1] tensors plus lengths [1, P]."""
    n_paths = len(paths)
    times = torch.full((1, n_paths, MAX_EVENTS, 1), PAD_TIME)
    marks = torch.zeros((1, n_paths, MAX_EVENTS, 1), dtype=torch.long)
    lengths = torch.zeros((1, n_paths), dtype=torch.long)
    for i, (path_times, path_marks) in enumerate(paths):
        n = len(path_times)
        times[0, i, :n, 0] = torch.from_numpy(path_times)
        marks[0, i, :n, 0] = torch.from_numpy(path_marks.astype(np.int64))
        lengths[0, i] = n
    return times, marks, lengths


def person_path(events: pd.DataFrame, before: pd.Timestamp | None) -> tuple[np.ndarray, np.ndarray] | None:
    """A person's last MAX_EVENTS events before a date, as days since their first kept event."""
    if before is not None:
        events = events[events["timestamp"] < before]
    events = events.tail(MAX_EVENTS)
    if events.empty:
        return None
    return to_days(events["timestamp"], events["timestamp"].iloc[0]), events["mark"].to_numpy()


def context_for(patient_id: str, sequences: pd.DataFrame, anchors: pd.DataFrame, split: str) -> list:
    """Other people's timelines: before the cut date (time split) or from other folds (patient split)."""
    if split == "time":
        allowed = sequences[(sequences["patient_id"] != patient_id) & (sequences["timestamp"] < TIME_SPLIT_DATE)]
    else:
        own_fold = anchors.loc[anchors["patient_id"] == patient_id, "fold"].iloc[0]
        other = anchors.loc[anchors["fold"] != own_fold, "patient_id"].unique()
        allowed = sequences[sequences["patient_id"].isin(other)]
    paths = [person_path(grp, None) for _, grp in allowed.groupby("patient_id")]
    return [p for p in paths if p is not None and len(p[0]) >= 2]


def score_person(model, person: pd.DataFrame, person_anchors: pd.Series, context: list) -> pd.DataFrame:
    """Expected crisis count and expected total events in the next 28 days, for each anchor."""
    inference, offsets, keep = [], [], []
    for anchor in person_anchors:
        path = person_path(person, anchor)
        if path is None:
            continue
        first_kept = person[person["timestamp"] < anchor].tail(MAX_EVENTS)["timestamp"].iloc[0]
        inference.append(path)
        offsets.append(float((anchor - first_kept) / pd.Timedelta(days=1)))
        keep.append(anchor)
    result = pd.DataFrame({"anchor": person_anchors, "fimpp_crisis_28d": np.nan, "fimpp_events_28d": np.nan})
    if not inference:
        return result

    ctx_times, ctx_marks, ctx_lengths = pad_paths(context)
    inf_times, inf_marks, inf_lengths = pad_paths(inference)
    grid = torch.arange(0, HORIZON_DAYS + 1, dtype=torch.float32)
    eval_times = torch.tensor(offsets).view(1, -1, 1) + grid.view(1, 1, -1)
    batch = {
        "context_event_times": ctx_times,
        "context_event_types": ctx_marks,
        "context_seq_lengths": ctx_lengths,
        "inference_event_times": inf_times,
        "inference_event_types": inf_marks,
        "inference_seq_lengths": inf_lengths,
        "intensity_evaluation_times": eval_times,
        "num_marks": NUM_MARKS,
    }
    with torch.no_grad():
        intensity = model(batch)["predicted_intensity_values"][0, :, :, : len(grid)]  # [M, P, T]
    crisis = torch.trapezoid(intensity[CRISIS_MARK], grid, dim=-1).numpy()
    total = torch.trapezoid(intensity.sum(dim=0), grid, dim=-1).numpy()
    scored = pd.DataFrame({"anchor": keep, "fimpp_crisis_28d": crisis, "fimpp_events_28d": total})
    return result[["anchor"]].merge(scored, on="anchor", how="left")


REMOTE_CODE_FILES = ("__init__.py", "configuration_hawkes.py", "modeling_hawkes.py")


def load_model(model_dir: Path | None):
    """Pretrained FIM-PP from the Hub, or a fine-tuned checkpoint folder (remote-code files copied in)."""
    hub_dir = Path(snapshot_download(repo_id="FIM4Science/fim-pp"))
    if model_dir is None:
        model_dir = hub_dir
    else:
        for name in REMOTE_CODE_FILES:
            if not (model_dir / name).exists():
                shutil.copy(hub_dir / name, model_dir / name)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    return model.eval()


def main() -> None:
    """Score every anchor for the chosen split and write fimpp_scores_<split>.parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["time", "patient"], required=True)
    parser.add_argument("--model-dir", type=Path, default=None, help="fine-tuned checkpoint folder")
    parser.add_argument("--tag", default="", help="suffix for the output file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    torch.manual_seed(0)

    sequences = pd.read_parquet(EXPORT_DIR / "sequences.parquet")
    anchors = pd.read_parquet(EXPORT_DIR / "anchors.parquet")
    model = load_model(args.model_dir)

    results = []
    for patient_id, person_anchors in anchors.groupby("patient_id"):
        person = sequences[sequences["patient_id"] == patient_id]
        context = context_for(patient_id, sequences, anchors, args.split)
        scores = score_person(model, person, person_anchors["anchor"], context)
        scores.insert(0, "patient_id", patient_id)
        results.append(scores)
        logger.info("%s: %d anchors, %d context paths", patient_id, len(scores), len(context))

    out = pd.concat(results, ignore_index=True)
    out.to_parquet(EXPORT_DIR / f"fimpp_scores_{args.split}{args.tag}.parquet")
    logger.info("Wrote %d scores (%d missing)", len(out), out["fimpp_crisis_28d"].isna().sum())


if __name__ == "__main__":
    main()
