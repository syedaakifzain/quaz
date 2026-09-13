"""Model artifact serialization and loading.

Phase 12: Save and load all model artifacts required for inference.

Artifacts saved:
    - vqc_model.dill — trained VQC
    - pre_scaler.joblib — fitted StandardScaler
    - pca.joblib — fitted PCA transformer
    - scaler.joblib — fitted MinMaxScaler
    - feature_config.json — feature extraction configuration
    - preprocessing_config.json — preprocessing parameters
    - model_metadata.json — model version, metrics, provenance
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import dill
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from core.config import FeatureConfig, VQCConfig
from core.constants import MODEL_ARTIFACT_FILES

logger = logging.getLogger(__name__)


def save_model_artifacts(
    artifacts_dir: Path,
    vqc,
    pre_scaler: StandardScaler,
    pca: PCA,
    scaler: MinMaxScaler,
    feature_config: FeatureConfig,
    vqc_config: VQCConfig,
    model_version: str,
    training_subjects: list[str],
    threshold: float,
    best_ratio: float,
    evaluation_metrics: dict | None = None,
) -> Path:
    """Save all model artifacts to disk."""
    version_dir = artifacts_dir / model_version.replace(".", "_")
    version_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save VQC model
    vqc_path = version_dir / "vqc_model.dill"
    with open(vqc_path, "wb") as f:
        dill.dump(vqc, f)
    logger.info("Saved VQC model: %s", vqc_path)

    # 1.5 Save Pre-scaler
    pre_scaler_path = version_dir / "pre_scaler.joblib"
    joblib.dump(pre_scaler, pre_scaler_path)
    logger.info("Saved pre-scaler: %s", pre_scaler_path)

    # 2. Save PCA
    pca_path = version_dir / "pca.joblib"
    joblib.dump(pca, pca_path)
    logger.info("Saved PCA: %s", pca_path)

    # 3. Save scaler
    scaler_path = version_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info("Saved scaler: %s", scaler_path)

    # 4. Feature config
    feature_config_dict = {
        "channels": feature_config.channels,
        "features": feature_config.features,
        "segment_duration_seconds": feature_config.segment_duration_seconds,
        "wavelet_name": feature_config.wavelet.name,
        "wavelet_level": feature_config.wavelet.level,
        "delta_band_low_hz": feature_config.delta_band.low_hz,
        "delta_band_high_hz": feature_config.delta_band.high_hz,
        "pca_n_components": feature_config.pca.n_components,
        "scaler_type": feature_config.scaler.type,
        "scaler_range_min": feature_config.scaler.feature_range_min,
        "scaler_range_max": feature_config.scaler.feature_range_max,
        "total_features": feature_config.total_features,
        "schema_hash": feature_config.schema_hash(),
    }
    feat_path = version_dir / "feature_config.json"
    feat_path.write_text(json.dumps(feature_config_dict, indent=2), encoding="utf-8")
    logger.info("Saved feature config: %s", feat_path)

    # 5. Preprocessing config
    preproc_dict = {
        "filter_low_hz": feature_config.filter.low_hz,
        "filter_high_hz": feature_config.filter.high_hz,
        "segment_duration_seconds": feature_config.segment_duration_seconds,
        "channels": feature_config.channels,
    }
    preproc_path = version_dir / "preprocessing_config.json"
    preproc_path.write_text(json.dumps(preproc_dict, indent=2), encoding="utf-8")
    logger.info("Saved preprocessing config: %s", preproc_path)

    # 6. Model metadata
    metadata = {
        "model_name": "quantum_eeg_vqc",
        "model_version": model_version,
        "classifier": "VQC",
        "qubits": pca.n_components,  # Dynamic: matches selected PCA dimension
        "threshold": threshold,
        "best_ratio": best_ratio,
        "selected_pca_dim": pca.n_components,
        "feature_count_before_pca": feature_config.total_features,
        "feature_count_after_pca": pca.n_components,
        "channels": feature_config.channels,
        "segment_duration_seconds": feature_config.segment_duration_seconds,
        "filter_low_hz": feature_config.filter.low_hz,
        "filter_high_hz": feature_config.filter.high_hz,
        "feature_schema_hash": feature_config.schema_hash(),
        "training_subjects": training_subjects,
        "vqc_config": {
            "feature_map": vqc_config.feature_map,
            "feature_map_reps": vqc_config.feature_map_reps,
            "ansatz": vqc_config.ansatz,
            "ansatz_reps": vqc_config.ansatz_reps,
            "entanglement": vqc_config.entanglement,
            "optimizer": vqc_config.optimizer,
            "maxiter": vqc_config.maxiter,
        },
        "pca_explained_variance_ratio": [
            float(v) for v in pca.explained_variance_ratio_
        ],
        "pca_total_explained_variance": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "evaluation_metrics": evaluation_metrics or {},
        "created_at": datetime.now().isoformat(),
    }
    meta_path = version_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved model metadata: %s", meta_path)

    logger.info("All model artifacts saved to: %s", version_dir)
    return version_dir


def load_model_artifacts(
    artifacts_dir: Path,
    model_version: str,
) -> dict:
    """Load model artifacts for inference."""
    version_dir = artifacts_dir / model_version.replace(".", "_")
    if not version_dir.exists():
        raise FileNotFoundError(
            f"Model artifact directory not found: {version_dir}"
        )

    # Check all required files
    missing = [
        f for f in MODEL_ARTIFACT_FILES
        if not (version_dir / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts in {version_dir}: {missing}"
        )

    # Load VQC with fallback compatibility unpickler for Qiskit 2.5
    vqc_path = version_dir / "vqc_model.dill"
    try:
        with open(vqc_path, "rb") as f:
            vqc = dill.load(f)
    except Exception as e:
        logger.info("Standard dill load failed for %s (%s); trying QiskitVQCUnpickler fallback", vqc_path, e)
        import pickle
        import qiskit._accelerate.circuit as qac

        class QiskitVQCUnpickler(pickle._Unpickler):
            def load_newobj(self):
                args = self.stack.pop()
                cls = self.stack.pop()
                if cls is qac.ParameterExpression:
                    class PEPlaceholder: pass
                    self.stack.append(PEPlaceholder())
                    return
                obj = cls.__new__(cls, *args)
                self.stack.append(obj)

            def load_build(self):
                state = self.stack.pop()
                inst = self.stack[-1]
                if inst.__class__.__name__ == 'PEPlaceholder':
                    real_pe = qac.ParameterExpression._reconstruct(state)
                    self.stack[-1] = real_pe
                    return
                if hasattr(inst, '__setstate__'):
                    inst.__setstate__(state)
                elif isinstance(state, dict):
                    if hasattr(inst, '__dict__'):
                        inst.__dict__.update(state)
                elif isinstance(state, tuple):
                    if hasattr(inst, '__dict__') and state[0] and isinstance(state[0], dict):
                        inst.__dict__.update(state[0])
                    if len(state) > 1 and isinstance(state[1], dict):
                        for k, v in state[1].items():
                            try: setattr(inst, k, v)
                            except Exception: pass

        QiskitVQCUnpickler.dispatch = dict(pickle._Unpickler.dispatch)
        QiskitVQCUnpickler.dispatch[pickle.NEWOBJ[0]] = QiskitVQCUnpickler.load_newobj
        QiskitVQCUnpickler.dispatch[pickle.BUILD[0]] = QiskitVQCUnpickler.load_build

        with open(vqc_path, "rb") as f:
            vqc = QiskitVQCUnpickler(f).load()

    # Load pre-scaler
    pre_scaler = joblib.load(version_dir / "pre_scaler.joblib")

    # Load PCA
    pca = joblib.load(version_dir / "pca.joblib")

    # Load scaler
    scaler = joblib.load(version_dir / "scaler.joblib")

    # Load configs
    feature_config = json.loads(
        (version_dir / "feature_config.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (version_dir / "model_metadata.json").read_text(encoding="utf-8")
    )

    logger.info("Loaded model artifacts from: %s", version_dir)

    return {
        "vqc": vqc,
        "pre_scaler": pre_scaler,
        "pca": pca,
        "scaler": scaler,
        "feature_config": feature_config,
        "metadata": metadata,
    }
