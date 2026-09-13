"""Training pipeline checkpointing and resume module.

Manages atomic fold-level and production-level checkpoint serialization,
validation, atomic writing, and resumption for the Quantum EEG Signal Classifier.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import dill
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from core.config import FeatureConfig, VQCConfig

logger = logging.getLogger(__name__)

REQUIRED_FOLD_FILES = [
    "vqc_model.dill",
    "pre_scaler.joblib",
    "pca.joblib",
    "scaler.joblib",
    "fold_metadata.json",
]

REQUIRED_PROD_FILES = [
    "vqc_model.dill",
    "pre_scaler.joblib",
    "pca.joblib",
    "scaler.joblib",
    "model_metadata.json",
    "production_metadata.json",
]


def get_checkpoint_dir(artifacts_dir: Path, model_version: str) -> Path:
    """Return the checkpoint directory for a model version.
    
    Example: models/v1_3_0/checkpoints
    """
    version_dir_name = model_version.replace(".", "_")
    return artifacts_dir / version_dir_name / "checkpoints"


def get_version_dir(artifacts_dir: Path, model_version: str) -> Path:
    """Return the root artifact directory for a model version.
    
    Example: models/v1_3_0
    """
    version_dir_name = model_version.replace(".", "_")
    return artifacts_dir / version_dir_name


# ──────────────────────────────────────────────────────────────
# Fold Checkpointing
# ──────────────────────────────────────────────────────────────

def save_fold_checkpoint(
    checkpoint_dir: Path,
    fold_index: int,
    test_subject: str,
    train_subjects: list[str],
    vqc: Any,
    pre_scaler: StandardScaler,
    pca: PCA,
    scaler: MinMaxScaler,
    metrics: dict[str, Any],
    selected_pca_dim: int,
    selected_ratio: float,
    selected_threshold: float,
    model_version: str,
    feature_schema_hash: str,
) -> Path:
    """Atomically save a completed fold's trained artifacts and metadata."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    target_dir = checkpoint_dir / f"fold_{fold_index}"
    temp_dir = checkpoint_dir / f"fold_{fold_index}_tmp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Save VQC
        with open(temp_dir / "vqc_model.dill", "wb") as f:
            dill.dump(vqc, f)

        # 2. Save pre-scaler, PCA, scaler
        joblib.dump(pre_scaler, temp_dir / "pre_scaler.joblib")
        joblib.dump(pca, temp_dir / "pca.joblib")
        joblib.dump(scaler, temp_dir / "scaler.joblib")

        # 3. Save fold metadata
        fold_metadata = {
            "fold_index": fold_index,
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "selected_pca_dim": selected_pca_dim,
            "selected_ratio": selected_ratio,
            "selected_threshold": selected_threshold,
            "metrics": metrics,
            "model_version": model_version,
            "feature_schema_hash": feature_schema_hash,
            "status": "COMPLETED",
            "saved_at": datetime.now().isoformat(),
        }
        meta_path = temp_dir / "fold_metadata.json"
        meta_path.write_text(json.dumps(fold_metadata, indent=2), encoding="utf-8")

        # Atomic directory swap
        if target_dir.exists():
            shutil.rmtree(target_dir)
        temp_dir.rename(target_dir)
        logger.info("Fold %d checkpoint saved atomically to: %s", fold_index, target_dir)
        return target_dir

    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        logger.error("Failed to write fold %d checkpoint: %s", fold_index, e)
        raise


def load_fold_checkpoint(checkpoint_dir: Path, fold_index: int) -> dict[str, Any] | None:
    """Validate and load a completed fold checkpoint.
    
    Returns dict with fold metadata and artifact objects, or None if invalid.
    """
    target_dir = checkpoint_dir / f"fold_{fold_index}"
    if not target_dir.exists() or not target_dir.is_dir():
        return None

    for fname in REQUIRED_FOLD_FILES:
        fpath = target_dir / fname
        if not fpath.exists() or fpath.stat().st_size == 0:
            logger.warning("Fold %d checkpoint incomplete: missing or empty %s", fold_index, fname)
            return None

    try:
        meta_text = (target_dir / "fold_metadata.json").read_text(encoding="utf-8")
        meta = json.loads(meta_text)
        if meta.get("status") != "COMPLETED":
            logger.warning("Fold %d checkpoint status is not COMPLETED", fold_index)
            return None

        # Verify artifacts load cleanly
        with open(target_dir / "vqc_model.dill", "rb") as f:
            vqc = dill.load(f)

        pre_scaler = joblib.load(target_dir / "pre_scaler.joblib")
        pca = joblib.load(target_dir / "pca.joblib")
        scaler = joblib.load(target_dir / "scaler.joblib")

        return {
            "metadata": meta,
            "vqc": vqc,
            "pre_scaler": pre_scaler,
            "pca": pca,
            "scaler": scaler,
        }
    except Exception as e:
        logger.warning("Failed to load fold %d checkpoint from %s: %s", fold_index, target_dir, e)
        return None


def get_completed_folds(checkpoint_dir: Path, total_folds: int) -> list[int]:
    """Return sorted list of valid completed fold indices."""
    completed = []
    for idx in range(total_folds):
        ckpt = load_fold_checkpoint(checkpoint_dir, idx)
        if ckpt is not None:
            completed.append(idx)
    return sorted(completed)


# ──────────────────────────────────────────────────────────────
# Production Model Checkpointing
# ──────────────────────────────────────────────────────────────

def save_production_checkpoint(
    checkpoint_dir: Path,
    vqc: Any,
    pre_scaler: StandardScaler,
    pca: PCA,
    scaler: MinMaxScaler,
    metadata: dict[str, Any],
    eval_report: dict[str, Any],
    version_dir: Path,
) -> Path:
    """Atomically save the completed production model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prod_target = checkpoint_dir / "production"
    prod_temp = checkpoint_dir / "production_tmp"

    if prod_temp.exists():
        shutil.rmtree(prod_temp)
    prod_temp.mkdir(parents=True, exist_ok=True)

    try:
        # Save artifacts to temp checkpoint directory
        with open(prod_temp / "vqc_model.dill", "wb") as f:
            dill.dump(vqc, f)

        joblib.dump(pre_scaler, prod_temp / "pre_scaler.joblib")
        joblib.dump(pca, prod_temp / "pca.joblib")
        joblib.dump(scaler, prod_temp / "scaler.joblib")

        (prod_temp / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        
        prod_meta = {
            "model_version": metadata.get("model_version"),
            "status": "COMPLETED",
            "completed_at": datetime.now().isoformat(),
        }
        (prod_temp / "production_metadata.json").write_text(
            json.dumps(prod_meta, indent=2), encoding="utf-8"
        )

        # Atomic directory swap
        if prod_target.exists():
            shutil.rmtree(prod_target)
        prod_temp.rename(prod_target)
        logger.info("Production model checkpoint saved atomically to: %s", prod_target)
        return prod_target

    except Exception as e:
        if prod_temp.exists():
            shutil.rmtree(prod_temp, ignore_errors=True)
        logger.error("Failed to write production model checkpoint: %s", e)
        raise


def load_production_checkpoint(
    checkpoint_dir: Path,
    version_dir: Path,
) -> dict[str, Any] | None:
    """Validate and load completed production model checkpoint.
    
    Returns dict if valid and COMPLETED, else None.
    """
    prod_target = checkpoint_dir / "production"
    if not prod_target.exists() or not prod_target.is_dir():
        return None

    for fname in REQUIRED_PROD_FILES:
        fpath = prod_target / fname
        if not fpath.exists() or fpath.stat().st_size == 0:
            logger.warning("Production checkpoint incomplete: missing or empty %s", fname)
            return None

    # Check main version directory evaluation report as well
    report_path = version_dir / "evaluation_report.json"
    if not report_path.exists() or report_path.stat().st_size == 0:
        logger.warning("Production checkpoint incomplete: missing or empty evaluation_report.json in %s", version_dir)
        return None

    try:
        prod_meta_text = (prod_target / "production_metadata.json").read_text(encoding="utf-8")
        prod_meta = json.loads(prod_meta_text)
        if prod_meta.get("status") != "COMPLETED":
            return None

        # Verify artifacts load cleanly
        with open(prod_target / "vqc_model.dill", "rb") as f:
            vqc = dill.load(f)

        pre_scaler = joblib.load(prod_target / "pre_scaler.joblib")
        pca = joblib.load(prod_target / "pca.joblib")
        scaler = joblib.load(prod_target / "scaler.joblib")

        metadata = json.loads((prod_target / "model_metadata.json").read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))

        return {
            "vqc": vqc,
            "pre_scaler": pre_scaler,
            "pca": pca,
            "scaler": scaler,
            "metadata": metadata,
            "evaluation_report": report,
            "production_metadata": prod_meta,
        }
    except Exception as e:
        logger.warning("Failed to load production checkpoint: %s", e)
        return None


def get_production_status(checkpoint_dir: Path, version_dir: Path) -> str:
    """Return status string for production model: COMPLETED, IN_PROGRESS, or NOT_STARTED."""
    prod_ckpt = load_production_checkpoint(checkpoint_dir, version_dir)
    if prod_ckpt is not None:
        return "COMPLETED"

    prod_target = checkpoint_dir / "production"
    prod_temp = checkpoint_dir / "production_tmp"
    if prod_target.exists() or prod_temp.exists():
        return "IN_PROGRESS"

    return "NOT_STARTED"
