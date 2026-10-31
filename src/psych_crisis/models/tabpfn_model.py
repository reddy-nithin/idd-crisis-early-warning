"""TabPFN: pretrained tabular foundation model, used zero-shot and fine-tuned.

Version is always set explicitly (2.5 is the main model, 3.5 the comparison), because the
package default changes between releases. Runs locally on the Mac with the `tabpfn` package.
Never swap in `tabpfn-client`: it uploads rows to Prior Labs.
"""

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tabpfn.constants import ModelVersion

from psych_crisis.config import PROJECT_ROOT, RANDOM_SEED

logger = logging.getLogger(__name__)

# TABPFN_TOKEN lives in the git-ignored .env file, never in code.
load_dotenv(PROJECT_ROOT / ".env")

DEVICE = "cpu"  # data stays on the Mac; MPS is unreliable for TabPFN on 8 GB
N_ESTIMATORS = 4
# Cap on context rows so the Mac's 8 GB is enough. Every positive row is always kept.
MAX_CONTEXT_ROWS = 4000
FINETUNE_EPOCHS = 30
FINETUNE_LEARNING_RATE = 1e-5
FINETUNE_TIME_LIMIT_SECONDS = 45 * 60

FitPredict = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]


def subsample_context(X: pd.DataFrame, y: np.ndarray, max_rows: int = MAX_CONTEXT_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Keep all positive rows and a random sample of negatives, up to max_rows in total."""
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    n_negatives = max(0, min(len(negatives), max_rows - len(positives)))
    rng = np.random.default_rng(RANDOM_SEED)
    keep = np.sort(np.concatenate([positives, rng.choice(negatives, size=n_negatives, replace=False)]))
    if len(keep) < len(y):
        logger.info("Context subsampled: %d of %d rows (all %d positives kept)", len(keep), len(y), len(positives))
    return X.to_numpy(dtype=np.float32)[keep], y[keep]


def _zero_shot(version: ModelVersion, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    """Training rows become the model's context; no weights change."""
    from tabpfn import TabPFNClassifier

    model = TabPFNClassifier.create_default_for_version(
        version,
        n_estimators=N_ESTIMATORS,
        device=DEVICE,
        random_state=RANDOM_SEED,
        ignore_pretraining_limits=True,
    )
    X_context, y_context = subsample_context(X_train, y_train)
    model.fit(X_context, y_context)
    logger.info("TabPFN %s zero-shot: %d context rows", version.value, len(y_context))
    return model.predict_proba(X_test.to_numpy(dtype=np.float32))[:, 1]


def _finetuned(version: ModelVersion, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    """Fine-tune the weights on the training split (early stopping, time-capped), then predict."""
    from tabpfn.finetuning import FinetunedTabPFNClassifier

    model = FinetunedTabPFNClassifier(
        device=DEVICE,
        model_version=version,
        epochs=FINETUNE_EPOCHS,
        learning_rate=FINETUNE_LEARNING_RATE,
        time_limit=FINETUNE_TIME_LIMIT_SECONDS,
        validation_split_ratio=0.2,
        early_stopping=True,
        early_stopping_patience=5,
        random_state=RANDOM_SEED,
    )
    X_context, y_context = subsample_context(X_train, y_train)
    model.fit(X_context, y_context)
    logger.info("TabPFN %s fine-tuned on %d rows", version.value, len(y_context))
    return model.predict_proba(X_test.to_numpy(dtype=np.float32))[:, 1]


def tabpfn_fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    """TabPFN-2.5 zero-shot (main model)."""
    return _zero_shot(ModelVersion.V2_5, X_train, y_train, X_test)


def tabpfn35_fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    """TabPFN-3.5 zero-shot (comparison)."""
    return _zero_shot(ModelVersion.V3_5, X_train, y_train, X_test)


def tabpfn_finetuned_fit_predict(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    """TabPFN-2.5 fine-tuned."""
    return _finetuned(ModelVersion.V2_5, X_train, y_train, X_test)
