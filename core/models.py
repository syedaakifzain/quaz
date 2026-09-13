"""Domain data models for the Quantum EEG Signal Classifier.

All structured data passed between modules uses these typed dataclasses
instead of raw dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ── EEG Recording Metadata ────────────────────────────────────


@dataclass
class EEGMetadata:
    """Metadata extracted from an EDF file."""
    file_name: str
    file_path: str
    sampling_frequency: float
    duration_seconds: float
    channel_count: int
    channel_names: list[str]
    n_samples: int


# ── Segment Prediction ────────────────────────────────────────


@dataclass
class SegmentPrediction:
    """Prediction result for a single 2-second segment."""
    segment_index: int
    start_time_seconds: float
    end_time_seconds: float
    predicted_label: int  # 0 = non-seizure, 1 = seizure
    label_name: str  # "Non-Seizure" or "Seizure"
    seizure_probability: float = -1.0  # VQC positive-class probability


# ── Analysis Result ───────────────────────────────────────────


@dataclass
class AnalysisResult:
    """Complete result of an inference analysis on an uploaded EDF."""
    recording_name: str
    duration_seconds: float
    sampling_frequency: float
    channels_used: list[str]
    total_segments: int
    seizure_segments: int
    non_seizure_segments: int
    prediction_summary: str  # e.g. "Possible seizure activity detected"
    segment_predictions: list[SegmentPrediction]
    model_version: str
    processing_time_seconds: float
    warnings: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ── Model Manifest ────────────────────────────────────────────


@dataclass
class ModelManifest:
    """Information about a loaded model and its training provenance."""
    model_name: str
    model_version: str
    classifier: str
    qubits: int
    feature_count_before_pca: int
    feature_count_after_pca: int
    channels: list[str]
    segment_duration_seconds: float
    filter_low_hz: float
    filter_high_hz: float
    feature_schema_hash: str
    evaluation_metrics: dict[str, Any] = field(default_factory=dict)
    package_versions: dict[str, str] = field(default_factory=dict)
    created_at: str = ""


# ── Training Report Models ────────────────────────────────────


@dataclass
class SkippedFile:
    """Record of a file skipped during dataset building."""
    file: str
    subject: str
    reason: str


@dataclass
class FoldMetrics:
    """Evaluation metrics for one LOSO fold."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sensitivity: float
    specificity: float
    roc_auc: float | None  # None if probabilities unavailable
    confusion_matrix: list[list[int]]  # [[TN, FP], [FN, TP]]


@dataclass
class LOSOFoldResult:
    """Result from a single LOSO experiment."""
    fold: int
    train_subjects: list[str]
    test_subject: str
    train_segments_before_balance: int
    train_seizure_before_balance: int
    train_nonseizure_before_balance: int
    train_segments_after_balance: int
    test_segments: int
    test_seizure: int
    test_nonseizure: int
    metrics: FoldMetrics


@dataclass
class AggregateMetrics:
    """Mean ± std of metrics across LOSO folds."""
    accuracy: dict[str, float]  # {"mean": ..., "std": ...}
    precision: dict[str, float]
    recall: dict[str, float]
    f1_score: dict[str, float]
    sensitivity: dict[str, float]
    specificity: dict[str, float]
    roc_auc: dict[str, float | None]


@dataclass
class DatasetDiscovery:
    """Summary of dataset discovery phase."""
    subjects_configured: list[str]
    total_edf_files_discovered: int
    total_edf_files_processed: int
    total_edf_files_skipped: int
    skipped_files: list[SkippedFile]


@dataclass
class SegmentationSummary:
    """Summary of segmentation results."""
    segment_duration_seconds: float
    sampling_frequency_hz: float
    samples_per_segment: int
    total_segments: int
    seizure_segments: int
    non_seizure_segments: int


@dataclass
class TrainingReport:
    """Complete training report persisted alongside model artifacts."""
    report_version: str
    generated_at: str
    dataset_discovery: DatasetDiscovery
    segmentation: SegmentationSummary
    loso_folds: list[LOSOFoldResult]
    aggregate_metrics: AggregateMetrics
    production_model_subjects: list[str]
    production_model_total_balanced_samples: int
    pca_n_components: int
    pca_explained_variance_ratio: list[float]
    pca_total_explained_variance: float
    vqc_config: dict[str, Any]
    package_versions: dict[str, str]
