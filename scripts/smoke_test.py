"""Smoke tests for the full training pipeline.

Phase 14: Validates every stage of the pipeline using a small subset
of data before committing to expensive full VQC training.

Tests:
    1.  EDF loading
    2.  Annotation parsing
    3.  Channel validation
    4.  Filtering
    5.  Segmentation
    6.  Label generation
    7.  Feature extraction
    8.  Feature dimension = 20
    9.  Subject-aware split (LOSO)
    10. Pre-scaling and PCA transformation
    11. Scaling
    12. VQC construction

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_feature_config, load_training_config
from core.logging_setup import setup_logging

logger = logging.getLogger(__name__)

PASS = "[PASS]"
FAIL = "[FAIL]"


def run_smoke_tests() -> dict:
    """Run all smoke tests and return results."""
    setup_logging(level=logging.INFO)

    results: dict[str, str] = {}
    start = time.time()

    logger.info("=" * 60)
    logger.info("SMOKE TESTS -- Quantum EEG Signal Classifier")
    logger.info("=" * 60)

    # Load configs
    feature_config = load_feature_config()
    training_config = load_training_config()

    # ── Test 1: EDF Loading ──
    test_name = "1. EDF Loading"
    try:
        from eeg_processing.loader import load_edf, extract_metadata

        base_dir = PROJECT_ROOT / training_config.dataset.base_dir
        subject_id = training_config.dataset.subjects[0]
        subject_dir = base_dir / subject_id
        edf_files = sorted(subject_dir.glob("*.edf"))
        assert len(edf_files) > 0, f"No EDF files in {subject_dir}"

        test_edf = edf_files[0]
        raw = load_edf(test_edf, preload=False)
        meta = extract_metadata(raw, test_edf)
        assert meta.sampling_frequency > 0
        assert meta.channel_count > 0
        assert meta.duration_seconds > 0
        logger.info("%s: %s -- loaded %s (%.1f Hz, %.1fs, %d ch)",
                    PASS, test_name, test_edf.name,
                    meta.sampling_frequency, meta.duration_seconds, meta.channel_count)
        results[test_name] = PASS
    except Exception as e:
        logger.error("%s: %s -- %s", FAIL, test_name, e)
        results[test_name] = f"{FAIL}: {e}"

    # ── Test 2: Annotation Parsing ──
    test_name = "2. Annotation Parsing"
    try:
        from eeg_processing.annotations import discover_summary_file, parse_summary_file

        summary_path = discover_summary_file(subject_dir)
        assert summary_path is not None, f"No summary file in {subject_dir}"

        annotations = parse_summary_file(summary_path)
        assert annotations.subject_id is not None
        assert len(annotations.file_annotations) > 0

        n_with_seizure = sum(
            1 for fa in annotations.file_annotations.values()
            if fa.num_seizures > 0
        )
        logger.info("%s: %s -- %d files, %d with seizures",
                    PASS, test_name, len(annotations.file_annotations), n_with_seizure)
        results[test_name] = PASS
    except Exception as e:
        logger.error("%s: %s -- %s", FAIL, test_name, e)
        results[test_name] = f"{FAIL}: {e}"

    # ── Test 3: Channel Validation ──
    test_name = "3. Channel Validation"
    try:
        from eeg_processing.validator import validate_channels
        raw_test = load_edf(test_edf, preload=False)
        validate_channels(raw_test, feature_config)
        logger.info("%s: %s -- all %d channels found",
                    PASS, test_name, len(feature_config.channels))
        results[test_name] = PASS
    except Exception as e:
        logger.error("%s: %s -- %s", FAIL, test_name, e)
        results[test_name] = f"{FAIL}: {e}"

    # ── Test 4: Filtering ──
    test_name = "4. Bandpass Filtering"
    try:
        from eeg_processing.channel_selector import select_channels
        from eeg_processing.filtering import bandpass_filter

        raw_sel = select_channels(load_edf(test_edf, preload=False), feature_config.channels)
        raw_filt = bandpass_filter(raw_sel, feature_config.filter.low_hz, feature_config.filter.high_hz)
        assert raw_filt.get_data().shape[0] == len(feature_config.channels)
        logger.info("%s: %s -- %.1f-%.1f Hz",
                    PASS, test_name, feature_config.filter.low_hz, feature_config.filter.high_hz)
        results[test_name] = PASS
    except Exception as e:
        logger.error("%s: %s -- %s", FAIL, test_name, e)
        results[test_name] = f"{FAIL}: {e}"

    # ── Test 5: Segmentation ──
    test_name = "5. Segmentation"
    try:
        from eeg_processing.segmentation import segment_recording

        data = raw_filt.get_data()
        sfreq = raw_filt.info["sfreq"]
        segments = segment_recording(data, sfreq, feature_config.segment_duration_seconds)
        assert len(segments) > 0
        expected_samples = int(sfreq * feature_config.segment_duration_seconds)
        assert segments[0].data.shape == (len(feature_config.channels), expected_samples)
        logger.info("%s: %s -- %d segments of %d samples",
                    PASS, test_name, len(segments), expected_samples)
        results[test_name] = PASS
    except Exception as e:
        logger.error("%s: %s -- %s", FAIL, test_name, e)
        results[test_name] = f"{FAIL}: {e}"

    # ── Test 6: Label Generation ──
    test_name = "6. Label Generation"
    try:
        from eeg_processing.annotations import get_seizure_intervals
        from training.build_dataset import _label_segment

        seizure_intervals = get_seizure_intervals(annotations, test_edf.name)
        labels = [_label_segment(seg, seizure_intervals) for seg in segments[:10]]
        assert all(l in (0, 1) for l in labels)
        logger.info("%s: %s -- labeled 10 segments (seizure intervals=%d)",
                    PASS, test_name, len(seizure_intervals))
        results[test_name] = PASS
    except Exception as e:
        logger.error("%s: %s -- %s", FAIL, test_name, e)
        results[test_name] = f"{FAIL}: {e}"

    # ── Test 7 & 8: Feature Extraction & Dimension ──
    test_name = "7. Feature Extraction"
    try:
        from eeg_processing.feature_extraction import extract_features_for_segment

        feat = extract_features_for_segment(segments[0].data, sfreq, feature_config)
        assert feat.shape == (feature_config.total_features,), \
            f"Expected {feature_config.total_features}, got {feat.shape}"
        assert np.all(np.isfinite(feat))
        logger.info("%s: %s -- dim=%d", PASS, test_name, feat.shape[0])
        results[test_name] = PASS
    except Exception as e:
        logger.error("%s: %s -- %s", FAIL, test_name, e)
        results[test_name] = f"{FAIL}: {e}"

    test_name = "8. Feature Dimension = 20"
    try:
        assert feat.shape[0] == feature_config.total_features, \
            f"Expected {feature_config.total_features}, got {feat.shape[0]}"
        logger.info("%s: %s", PASS, test_name)
        results[test_name] = PASS
    except Exception as e:
        logger.error("%s: %s -- %s", FAIL, test_name, e)
        results[test_name] = f"{FAIL}: {e}"

    # ── Test 9: Subject-Aware Split (LOSO) ──
    test_name = "9. Subject-Aware Split (LOSO)"
    try:
        # Build a small dataset (1 file per subject)
        from training.build_dataset import build_dataset

        dataset = build_dataset(
            training_config, feature_config, PROJECT_ROOT,
            max_files_per_subject=1,
        )
        assert dataset.X.shape[1] == feature_config.total_features
        assert len(dataset.subjects) >= 2

        from training.prepare_dataset import create_loso_folds

        folds = create_loso_folds(dataset.X, dataset.y, dataset.metadata, dataset.subjects)
        assert len(folds) == len(dataset.subjects)

        # Verify no leakage
        for fold in folds:
            train_subjects = set(m.subject_id for m in fold.meta_train)
            test_subjects = set(m.subject_id for m in fold.meta_test)
            assert len(train_subjects & test_subjects) == 0, "Subject leakage detected!"

        logger.info("%s: %s -- %d folds, no leakage",
                    PASS, test_name, len(folds))
        results[test_name] = PASS
    except Exception as e:
        logger.error("%s: %s -- %s", FAIL, test_name, e)
        results[test_name] = f"{FAIL}: {e}"

    # ── Test 10: PCA Transformation ──
    test_name = "10. Pre-scaling and PCA Transformation"
    try:
        from training.prepare_dataset import fit_pre_scaler, fit_pca

        pre_scaler = fit_pre_scaler(dataset.X)
        X_prescaled = pre_scaler.transform(dataset.X)
        pca = fit_pca(X_prescaled, feature_config.pca.n_components)
        X_pca = pca.transform(X_prescaled)
        assert X_pca.shape == (dataset.X.shape[0], feature_config.pca.n_components)
        var_explained = float(np.sum(pca.explained_variance_ratio_))
        logger.info("%s: %s -- %d -> %d, variance explained=%.3f",
                    PASS, test_name, feature_config.total_features,
                    feature_config.pca.n_components, var_explained)
        results[test_name] = PASS
    except Exception as e:
        logger.error("%s: %s -- %s", FAIL, test_name, e)
        results[test_name] = f"{FAIL}: {e}"

    # ── Test 11: Scaling ──
    test_name = "11. Scaling"
    try:
        from training.prepare_dataset import fit_scaler

        scaler = fit_scaler(X_pca, feature_config.scaler.feature_range)
        X_scaled = scaler.transform(X_pca)
        assert X_scaled.shape == X_pca.shape
        # Check range approximately matches [0, π]
        assert X_scaled.min() >= -0.01, f"Min={X_scaled.min()}"
        assert X_scaled.max() <= feature_config.scaler.feature_range_max + 0.01
        logger.info("%s: %s -- range [%.3f, %.3f]",
                    PASS, test_name, X_scaled.min(), X_scaled.max())
        results[test_name] = PASS
    except Exception as e:
        logger.error("%s: %s -- %s", FAIL, test_name, e)
        results[test_name] = f"{FAIL}: {e}"

    # ── Test 12: VQC Construction (default 4 qubits) ──
    test_name = "12. VQC Construction (4 qubits)"
    try:
        from quantum.classifier import build_vqc

        vqc = build_vqc(training_config.vqc, num_qubits=4)
        assert vqc is not None
        logger.info("%s: %s -- 4-qubit VQC constructed",
                    PASS, test_name)
        results[test_name] = PASS
    except Exception as e:
        logger.error("%s: %s -- %s", FAIL, test_name, e)
        results[test_name] = f"{FAIL}: {e}"

    # ── Test 13: Dynamic Qubit VQC Construction (6 and 8 qubits) ──
    test_name = "13. Dynamic Qubit VQC Construction"
    try:
        for n_q in [6, 8]:
            vqc_dyn = build_vqc(training_config.vqc, num_qubits=n_q)
            assert vqc_dyn is not None, f"VQC construction failed for {n_q} qubits"
        logger.info("%s: %s -- 6-qubit and 8-qubit VQCs constructed",
                    PASS, test_name)
        results[test_name] = PASS
    except Exception as e:
        logger.error("%s: %s -- %s", FAIL, test_name, e)
        results[test_name] = f"{FAIL}: {e}"

    # ── Summary ──
    elapsed = time.time() - start
    n_pass = sum(1 for v in results.values() if v == PASS)
    n_total = len(results)

    logger.info("=" * 60)
    logger.info("SMOKE TEST SUMMARY: %d/%d passed (%.1fs)", n_pass, n_total, elapsed)
    logger.info("=" * 60)
    for name, status in results.items():
        logger.info("  %s: %s", name, status)

    # ── Dataset Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)

    # Count all EDF and annotation files
    base_dir = PROJECT_ROOT / training_config.dataset.base_dir
    total_edf = 0
    total_annotation = 0
    for subj in training_config.dataset.subjects:
        subj_dir = base_dir / subj
        if subj_dir.is_dir():
            edfs = list(subj_dir.glob("*.edf"))
            total_edf += len(edfs)
            summaries = list(subj_dir.glob("*-summary.txt"))
            seizure_files = list(subj_dir.glob("*.edf.seizures"))
            total_annotation += len(summaries) + len(seizure_files)

    logger.info("  Subjects detected: %d -- %s",
                len(training_config.dataset.subjects),
                training_config.dataset.subjects)
    logger.info("  Total EDF files: %d", total_edf)
    logger.info("  Annotation files: %d", total_annotation)
    logger.info("  Feature dimension: %d", feature_config.total_features)
    logger.info("")
    logger.info("  Full training command:")
    logger.info("    python scripts/train.py")
    logger.info("")

    return results


if __name__ == "__main__":
    results = run_smoke_tests()
    # Exit with failure code if any test failed
    if any(v != PASS for v in results.values()):
        sys.exit(1)
