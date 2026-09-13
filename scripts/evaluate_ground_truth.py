"""Ground-truth evaluation — compare v1.2.0 VQC predictions against CHB-MIT annotations.

Usage (single recording):
    python scripts/evaluate_ground_truth.py test_data/chb06_01.edf test_data/chb06-summary.txt

Usage (multiple recordings — aggregate automatically):
    python scripts/evaluate_ground_truth.py ^
        test_data/chb06_01.edf test_data/chb06-summary.txt ^
        test_data/chb07_01.edf test_data/chb07-summary.txt

Each EDF must be immediately followed by its summary file.

Ground-truth labelling rule
---------------------------
A 2-second segment [seg_start, seg_end) is labelled **seizure** (1) if
it has ANY overlap with an annotated seizure interval [sz_start, sz_end).
Formally:  seg_start < sz_end  AND  sz_start < seg_end.

This is the standard "any-overlap" criterion used in the CHB-MIT seizure
detection literature.  It is the most inclusive (sensitive) defensible
rule — a segment is considered seizure-positive if even a single sample
falls within a seizure annotation.

Does NOT retrain or modify anything.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from eeg_processing.annotations import parse_summary_file, get_seizure_intervals
from inference.pipeline import InferenceService

# ── Defaults ────────────────────────────────────────────────────
ARTIFACTS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_VERSION = "v1.2.0"

logger = logging.getLogger(__name__)


# ── Data structures ─────────────────────────────────────────────

@dataclass
class ConfusionCounts:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / max(self.total, 1)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def sensitivity(self) -> float:
        return self.recall

    @property
    def specificity(self) -> float:
        return self.tn / max(self.tn + self.fp, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-12)

    @property
    def actual_seizure(self) -> int:
        return self.tp + self.fn

    @property
    def actual_non_seizure(self) -> int:
        return self.tn + self.fp

    @property
    def predicted_seizure(self) -> int:
        return self.tp + self.fp

    @property
    def predicted_non_seizure(self) -> int:
        return self.tn + self.fn

    def __add__(self, other: ConfusionCounts) -> ConfusionCounts:
        return ConfusionCounts(
            tp=self.tp + other.tp,
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )


# ── Ground-truth segment labelling ─────────────────────────────

def label_segments_ground_truth(
    segment_starts: list[float],
    segment_ends: list[float],
    seizure_intervals: list,
) -> np.ndarray:
    """Label each segment as seizure (1) or non-seizure (0).

    Rule: a segment is seizure-positive if it has ANY temporal overlap
    with an annotated seizure interval.

        overlap iff  seg_start < sz_end  AND  sz_start < seg_end

    Args:
        segment_starts: Start time of each segment in seconds.
        segment_ends:   End time of each segment in seconds.
        seizure_intervals: List of SeizureInterval with .start_seconds, .end_seconds.

    Returns:
        1-D int array of ground-truth labels (0 or 1), one per segment.
    """
    n = len(segment_starts)
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        for sz in seizure_intervals:
            if segment_starts[i] < sz.end_seconds and sz.start_seconds < segment_ends[i]:
                labels[i] = 1
                break

    return labels


# ── Evaluation ──────────────────────────────────────────────────

def evaluate_single(
    service: InferenceService,
    edf_path: Path,
    summary_path: Path,
) -> ConfusionCounts:
    """Run inference on one EDF and compare with ground truth."""

    # 1. Parse ground-truth annotations
    subject_annots = parse_summary_file(summary_path)
    edf_basename = edf_path.name
    seizure_intervals = get_seizure_intervals(subject_annots, edf_basename)

    print(f"  Ground-truth seizure intervals for {edf_basename}:")
    if seizure_intervals:
        for idx, si in enumerate(seizure_intervals, 1):
            print(f"    Seizure {idx}: {si.start_seconds:.0f}s – {si.end_seconds:.0f}s "
                  f"(duration {si.end_seconds - si.start_seconds:.0f}s)")
    else:
        print("    (none)")
    print()

    # 2. Run inference (uses existing pipeline exactly as-is)
    t0 = time.perf_counter()
    result = service.predict(edf_path)
    elapsed = time.perf_counter() - t0

    # 3. Build ground-truth labels for each segment
    seg_starts = [sp.start_time_seconds for sp in result.segment_predictions]
    seg_ends = [sp.end_time_seconds for sp in result.segment_predictions]
    y_true = label_segments_ground_truth(seg_starts, seg_ends, seizure_intervals)

    # 4. Probabilities
    probabilities = np.array(
        [sp.seizure_probability for sp in result.segment_predictions]
    )

    # 5. Current model decision at the configured threshold
    y_pred = (probabilities >= service.threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    counts = ConfusionCounts(tp=tp, tn=tn, fp=fp, fn=fn)

    # 6. Threshold sweep using the SAME VQC probabilities.
    # No retraining and no second inference.
    test_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]

    print()
    print("  THRESHOLD SWEEP (same VQC inference)")
    print("  " + "-" * 68)
    print(
        f"  {'Threshold':>9} {'TP':>6} {'FP':>6} {'FN':>6} "
        f"{'Precision':>11} {'Recall':>10} {'F1':>10}"
    )
    print("  " + "-" * 68)

    for threshold in test_thresholds:
        threshold_pred = (probabilities >= threshold).astype(int)

        t_tp = int(np.sum((y_true == 1) & (threshold_pred == 1)))
        t_tn = int(np.sum((y_true == 0) & (threshold_pred == 0)))
        t_fp = int(np.sum((y_true == 0) & (threshold_pred == 1)))
        t_fn = int(np.sum((y_true == 1) & (threshold_pred == 0)))

        threshold_counts = ConfusionCounts(
            tp=t_tp,
            tn=t_tn,
            fp=t_fp,
            fn=t_fn,
        )

        print(
            f"  {threshold:>9.2f} "
            f"{t_tp:>6} "
            f"{t_fp:>6} "
            f"{t_fn:>6} "
            f"{threshold_counts.precision:>11.6f} "
            f"{threshold_counts.recall:>10.6f} "
            f"{threshold_counts.f1:>10.6f}"
        )

    print("  " + "-" * 68)

    # 7. Print per-recording report
    print(f"  Inference time: {elapsed:.2f}s")
    print(f"  Total segments: {counts.total}")
    print()
    print(f"  Confusion matrix:")
    print(f"    TP (seizure correctly detected):    {counts.tp}")
    print(f"    TN (non-seizure correctly rejected):{counts.tn}")
    print(f"    FP (false alarm):                   {counts.fp}")
    print(f"    FN (missed seizure):                {counts.fn}")
    print()
    print(f"  Actual seizure segments:              {counts.actual_seizure}")
    print(f"  Actual non-seizure segments:           {counts.actual_non_seizure}")
    print(f"  Predicted seizure segments:            {counts.predicted_seizure}")
    print(f"  Predicted non-seizure segments:        {counts.predicted_non_seizure}")
    print()
    print(f"  Accuracy:                             {counts.accuracy:.6f}")
    print(f"  Precision:                            {counts.precision:.6f}")
    print(f"  Recall / Sensitivity:                 {counts.recall:.6f}")
    print(f"  Specificity:                          {counts.specificity:.6f}")
    print(f"  F1 score:                             {counts.f1:.6f}")
    print()
    print(f"  Min VQC seizure probability:          {probabilities.min():.6f}")
    print(f"  Max VQC seizure probability:          {probabilities.max():.6f}")
    print(f"  Mean VQC seizure probability:         {probabilities.mean():.6f}")
    print(f"  Binary decision threshold:            {service.threshold}")

    return counts


def print_aggregate(counts: ConfusionCounts, n_recordings: int) -> None:
    """Print aggregate metrics across all recordings."""
    print()
    print("=" * 72)
    print(f"AGGREGATE METRICS ({n_recordings} recordings)")
    print("=" * 72)
    print()
    print(f"  Total segments:                       {counts.total}")
    print(f"  TP:                                   {counts.tp}")
    print(f"  TN:                                   {counts.tn}")
    print(f"  FP:                                   {counts.fp}")
    print(f"  FN:                                   {counts.fn}")
    print()
    print(f"  Actual seizure segments:              {counts.actual_seizure}")
    print(f"  Actual non-seizure segments:           {counts.actual_non_seizure}")
    print(f"  Predicted seizure segments:            {counts.predicted_seizure}")
    print(f"  Predicted non-seizure segments:        {counts.predicted_non_seizure}")
    print()
    print(f"  Accuracy:                             {counts.accuracy:.6f}")
    print(f"  Precision:                            {counts.precision:.6f}")
    print(f"  Recall / Sensitivity:                 {counts.recall:.6f}")
    print(f"  Specificity:                          {counts.specificity:.6f}")
    print(f"  F1 score:                             {counts.f1:.6f}")
    print()
    print("=" * 72)


# ── CLI ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the saved VQC model against CHB-MIT ground-truth annotations. "
            "Supply one or more (EDF, summary) pairs as positional arguments."
        ),
    )
    parser.add_argument(
        "files",
        nargs="+",
        help=(
            "Alternating EDF and summary file paths.  "
            "Each EDF must be immediately followed by its summary file.  "
            "Example: chb06_01.edf chb06-summary.txt chb07_01.edf chb07-summary.txt"
        ),
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=DEFAULT_MODEL_VERSION,
        help=f"Model version to load (default: {DEFAULT_MODEL_VERSION}).",
    )
    return parser.parse_args()


def validate_pairs(files: list[str]) -> list[tuple[Path, Path]]:
    """Parse and validate alternating (EDF, summary) pairs."""
    if len(files) % 2 != 0:
        sys.exit(
            "ERROR: Arguments must be alternating EDF and summary file paths.\n"
            "       Got an odd number of arguments."
        )

    pairs: list[tuple[Path, Path]] = []
    for i in range(0, len(files), 2):
        edf = Path(files[i]).resolve()
        summary = Path(files[i + 1]).resolve()

        if not edf.exists():
            sys.exit(f"ERROR: EDF file not found: {edf}")
        if edf.suffix.lower() != ".edf":
            sys.exit(f"ERROR: Not an EDF file: {edf}")
        if not summary.exists():
            sys.exit(f"ERROR: Summary file not found: {summary}")
        if not summary.name.endswith("-summary.txt"):
            sys.exit(f"ERROR: Not a CHB-MIT summary file: {summary}")

        pairs.append((edf, summary))

    return pairs


def main() -> None:
    args = parse_args()
    pairs = validate_pairs(args.files)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    print("=" * 72)
    print("QUANTUM EEG SIGNAL CLASSIFIER — GROUND-TRUTH EVALUATION")
    print(f"  Model: {args.model_version}")
    print(f"  Recordings: {len(pairs)}")
    print()
    print("  Ground-truth labelling rule:")
    print("    A 2s segment is seizure-positive if it has ANY temporal overlap")
    print("    with an annotated seizure interval (any-overlap criterion).")
    print("=" * 72)
    print()

    service = InferenceService(ARTIFACTS_DIR, args.model_version)

    aggregate = ConfusionCounts()
    for idx, (edf_path, summary_path) in enumerate(pairs, 1):
        print("-" * 72)
        print(f"RECORDING {idx}: {edf_path.name}")
        print(f"  Summary:  {summary_path.name}")
        print("-" * 72)
        print()

        counts = evaluate_single(service, edf_path, summary_path)
        aggregate = aggregate + counts
        print()

    if len(pairs) > 1:
        print_aggregate(aggregate, len(pairs))


if __name__ == "__main__":
    main()
