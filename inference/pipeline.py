"""Inference pipeline — accepts an uploaded EDF and returns predictions.

Phase 15: Uses the SAME preprocessing, segmentation, and feature extraction
as training. Loads SAVED PCA, scaler, and VQC. NEVER retrains anything.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from core.config import FeatureConfig, load_feature_config
from core.models import AnalysisResult, SegmentPrediction
from eeg_processing.feature_extraction import extract_features_for_segment
from eeg_processing.pipeline import preprocess_edf
from quantum.classifier import predict_proba_vqc, predict_vqc
from training.save_artifacts import load_model_artifacts

logger = logging.getLogger(__name__)


class InferenceService:
    """Stateless inference service that loads model artifacts once.

    The service:
    - Loads the saved VQC, PCA, scalers once at init.
    - For each uploaded EDF:
        1. Runs the SAME preprocessing pipeline as training
        2. Extracts the SAME 20-dim features
        3. Applies pre_scaler.transform()
        4. Applies pca.transform() (NOT fit)
        5. Applies scaler.transform() (NOT fit)
        6. Predicts with the saved VQC using saved threshold
        7. Aggregates segment predictions
        8. Returns structured AnalysisResult
    """

    def __init__(
        self,
        artifacts_dir: Path,
        model_version: str,
        feature_config: FeatureConfig | None = None,
    ):
        """Load model artifacts.

        Args:
            artifacts_dir: Path to the models/ directory.
            model_version: Version string (e.g. "v1.0.0").
            feature_config: Optional override; defaults to loading from YAML.
        """
        self.artifacts = load_model_artifacts(artifacts_dir, model_version)
        self.vqc = self.artifacts["vqc"]
        self.pre_scaler = self.artifacts["pre_scaler"]
        self.pca = self.artifacts["pca"]
        self.scaler = self.artifacts["scaler"]
        self.metadata = self.artifacts["metadata"]
        self.threshold = self.metadata.get("threshold", 0.5)
        self.feature_config = feature_config or load_feature_config()
        self.model_version = model_version
        logger.info("InferenceService initialized (model %s, threshold %s)", model_version, self.threshold)

    def predict(self, edf_path: str | Path) -> AnalysisResult:
        """Run inference on an uploaded EDF file.

        Args:
            edf_path: Path to the EDF file.

        Returns:
            AnalysisResult with segment and recording-level predictions.
        """
        start_time = time.time()
        edf_path = Path(edf_path)

        # 1. Shared preprocessing (same as training)
        segments, sfreq = preprocess_edf(edf_path, self.feature_config)

        # 2. Extract features (same extractor)
        features_list = []
        for seg in segments:
            feat = extract_features_for_segment(
                seg.data, sfreq, self.feature_config,
            )
            features_list.append(feat)

        X = np.vstack(features_list)  # (n_segments, 20)

        # 3. Pre-scaler transform
        X_prescaled = self.pre_scaler.transform(X)

        # 4. PCA transform (NOT fit)
        X_pca = self.pca.transform(X_prescaled)

        # 5. Scaler transform (NOT fit)
        X_scaled = self.scaler.transform(X_pca)

        # 6. VQC predict — labels and probabilities
        y_pred = predict_vqc(self.vqc, X_scaled, threshold=self.threshold)
        probabilities = predict_proba_vqc(self.vqc, X_scaled)

        # 7. Build segment predictions
        segment_predictions: list[SegmentPrediction] = []
        for seg, label, prob in zip(segments, y_pred, probabilities):
            label_int = int(label)
            segment_predictions.append(SegmentPrediction(
                segment_index=seg.index,
                start_time_seconds=seg.start_seconds,
                end_time_seconds=seg.end_seconds,
                predicted_label=label_int,
                label_name="Seizure" if label_int == 1 else "Non-Seizure",
                seizure_probability=float(prob),
            ))

        # 8. Aggregate
        seizure_count = sum(1 for sp in segment_predictions if sp.predicted_label == 1)
        non_seizure_count = len(segment_predictions) - seizure_count

        if seizure_count > 0:
            prediction_summary = "Possible seizure activity detected"
        else:
            prediction_summary = "No seizure activity detected"

        # Recording-level seizure probability: mean VQC positive-class probability
        recording_seizure_probability = float(np.mean(probabilities))

        processing_time = time.time() - start_time

        result = AnalysisResult(
            recording_name=edf_path.name,
            duration_seconds=segments[-1].end_seconds if segments else 0.0,
            sampling_frequency=sfreq,
            channels_used=self.feature_config.channels,
            total_segments=len(segment_predictions),
            seizure_segments=seizure_count,
            non_seizure_segments=non_seizure_count,
            prediction_summary=prediction_summary,
            segment_predictions=segment_predictions,
            model_version=self.model_version,
            processing_time_seconds=processing_time,
        )

        logger.info(
            "Inference complete for %s: %d segments, %d seizure, "
            "recording seizure prob=%.4f, %.1fs",
            edf_path.name, len(segment_predictions), seizure_count,
            recording_seizure_probability, processing_time,
        )

        return result
