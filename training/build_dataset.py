"""Dataset builder — discover EDF files, preprocess, label, and extract features.

Phase 4 (Labeling): Assigns label=1 if a segment overlaps ANY seizure
interval by even 1 sample; otherwise label=0.

This module handles the full traversal of configured subjects and produces
the feature matrix X, label vector y, and metadata for each segment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.config import FeatureConfig, TrainingConfig
from core.constants import Label
from eeg_processing.annotations import (
    SeizureInterval,
    SubjectAnnotations,
    discover_summary_file,
    get_seizure_intervals,
    parse_summary_file,
)
from eeg_processing.feature_extraction import extract_features_for_segment
from eeg_processing.pipeline import preprocess_edf
from eeg_processing.segmentation import EEGSegment

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────


@dataclass
class SegmentMeta:
    """Provenance metadata for one training segment."""
    subject_id: str
    recording_id: str
    segment_index: int
    start_seconds: float
    end_seconds: float
    label: int


@dataclass
class DatasetResult:
    """Complete feature dataset ready for splitting and training."""
    X: np.ndarray                     # (n_segments, 16)
    y: np.ndarray                     # (n_segments,)
    metadata: list[SegmentMeta]       # one per segment
    subjects: list[str]               # unique subject IDs
    skipped: list[dict[str, str]]     # files that were skipped + reason


# ── Labeling ──────────────────────────────────────────────────


def _label_segment(
    segment: EEGSegment,
    seizure_intervals: list[SeizureInterval],
) -> int:
    """Assign a label to a segment.

    Overlap rule: label = 1 (SEIZURE) if the segment's time range
    [start, end) overlaps with any seizure interval [onset, offset).
    Otherwise label = 0 (NON_SEIZURE).

    Any non-zero overlap counts — even a single sample.
    """
    seg_start = segment.start_seconds
    seg_end = segment.end_seconds

    for interval in seizure_intervals:
        # Two ranges [a, b) and [c, d) overlap iff a < d and c < b
        if seg_start < interval.end_seconds and interval.start_seconds < seg_end:
            return Label.SEIZURE

    return Label.NON_SEIZURE


# ── Dataset Construction ─────────────────────────────────────


def build_dataset(
    training_config: TrainingConfig,
    feature_config: FeatureConfig,
    project_root: Path,
    max_files_per_subject: int | None = None,
) -> DatasetResult:
    """Build the complete training dataset from CHB-MIT EDF files.

    Args:
        training_config: Training configuration (subjects, paths, etc.).
        feature_config: Feature extraction / preprocessing configuration.
        project_root: Project root directory for resolving relative paths.
        max_files_per_subject: If set, limit EDF files per subject (for smoke tests).

    Returns:
        DatasetResult with feature matrix, labels, and metadata.
    """
    base_dir = project_root / training_config.dataset.base_dir
    subjects = training_config.dataset.subjects

    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    all_meta: list[SegmentMeta] = []
    skipped: list[dict[str, str]] = []
    seen_subjects: list[str] = []

    for subject_id in subjects:
        subject_dir = base_dir / subject_id
        if not subject_dir.is_dir():
            logger.warning("Subject directory not found: %s", subject_dir)
            skipped.append({"file": str(subject_dir), "subject": subject_id,
                            "reason": "subject directory not found"})
            continue

        # Parse annotations
        summary_path = discover_summary_file(subject_dir)
        if summary_path is None:
            logger.warning("No summary file in %s — skipping subject.", subject_dir)
            skipped.append({"file": str(subject_dir), "subject": subject_id,
                            "reason": "no summary file found"})
            continue

        annotations = parse_summary_file(summary_path)

        # Discover EDF files
        edf_files = sorted(subject_dir.glob("*.edf"))
        if max_files_per_subject is not None:
            edf_files = edf_files[:max_files_per_subject]

        if not edf_files:
            logger.warning("No EDF files in %s", subject_dir)
            skipped.append({"file": str(subject_dir), "subject": subject_id,
                            "reason": "no EDF files found"})
            continue

        seen_subjects.append(subject_id)

        for edf_path in edf_files:
            try:
                segments, sfreq = preprocess_edf(edf_path, feature_config)
            except Exception as e:
                logger.warning("Skipping %s: %s", edf_path.name, e)
                skipped.append({"file": edf_path.name, "subject": subject_id,
                                "reason": str(e)})
                continue

            seizure_intervals = get_seizure_intervals(annotations, edf_path.name)

            for seg in segments:
                label = _label_segment(seg, seizure_intervals)

                try:
                    feat = extract_features_for_segment(
                        seg.data, sfreq, feature_config,
                    )
                except Exception as e:
                    logger.warning(
                        "Feature extraction failed for %s seg %d: %s",
                        edf_path.name, seg.index, e,
                    )
                    continue

                all_X.append(feat)
                all_y.append(label)
                all_meta.append(SegmentMeta(
                    subject_id=subject_id,
                    recording_id=edf_path.name,
                    segment_index=seg.index,
                    start_seconds=seg.start_seconds,
                    end_seconds=seg.end_seconds,
                    label=label,
                ))

    if not all_X:
        raise RuntimeError("No segments were extracted — dataset is empty.")

    X = np.vstack(all_X)
    y = np.array(all_y, dtype=np.int64)

    n_seizure = int((y == Label.SEIZURE).sum())
    n_non = int((y == Label.NON_SEIZURE).sum())
    logger.info(
        "Dataset built: %d segments (%d seizure, %d non-seizure) from %d subjects",
        len(y), n_seizure, n_non, len(seen_subjects),
    )

    return DatasetResult(
        X=X, y=y, metadata=all_meta,
        subjects=seen_subjects, skipped=skipped,
    )
