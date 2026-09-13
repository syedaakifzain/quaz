"""Inference CLI — run the saved VQC model on any user-supplied EDF file.

Usage:
    python scripts/run_inference_test.py <path_to_edf>

Examples:
    python scripts/run_inference_test.py test_data/chb06_01.edf
    python scripts/run_inference_test.py "C:/path/to/any_recording.edf"

Does NOT train or modify anything.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from inference.pipeline import InferenceService

# ── Defaults ────────────────────────────────────────────────────
ARTIFACTS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_VERSION = "v1.3.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VQC seizure inference on an EDF recording.",
    )
    parser.add_argument(
        "edf_path",
        type=Path,
        help="Path to the EDF file to analyse.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=DEFAULT_MODEL_VERSION,
        help=f"Model version to load (default: {DEFAULT_MODEL_VERSION}).",
    )
    return parser.parse_args()


def validate_edf_path(edf_path: Path) -> Path:
    """Resolve and validate the supplied EDF path."""
    edf_path = edf_path.resolve()
    if not edf_path.exists():
        sys.exit(f"ERROR: File not found: {edf_path}")
    if not edf_path.is_file():
        sys.exit(f"ERROR: Not a file: {edf_path}")
    if edf_path.suffix.lower() != ".edf":
        sys.exit(f"ERROR: Not an EDF file (got '{edf_path.suffix}'): {edf_path}")
    return edf_path


def main() -> None:
    args = parse_args()
    edf_path = validate_edf_path(args.edf_path)
    model_version = args.model_version

    # Set up logging so we can see pipeline progress
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    print("=" * 72)
    print("QUANTUM EEG SIGNAL CLASSIFIER — INFERENCE")
    print(f"  Model: {model_version}")
    print(f"  EDF:   {edf_path}")
    print("=" * 72)
    print()

    # ── Load model & run inference ──────────────────────────────
    t0 = time.perf_counter()
    service = InferenceService(ARTIFACTS_DIR, model_version)
    result = service.predict(edf_path)
    total_time = time.perf_counter() - t0

    # ── Collect per-segment probabilities ───────────────────────
    probabilities = np.array(
        [sp.seizure_probability for sp in result.segment_predictions]
    )
    seizure_count = sum(
        1 for sp in result.segment_predictions if sp.predicted_label == 1
    )
    non_seizure_count = result.total_segments - seizure_count

    # ── Report ──────────────────────────────────────────────────
    pca_dim = service.metadata.get("feature_count_after_pca", "N/A")
    print()
    print("=" * 72)
    print("INFERENCE RESULTS")
    print("=" * 72)
    print()
    print(f"  Recording:                          {edf_path.name}")
    print(f"  Total segments:                     {result.total_segments}")
    print(f"  Feature dimension:                  20 (4 channels x 5 features)")
    print(f"  PCA output dimension:               {pca_dim}")
    print(f"  Predicted seizure segments:          {seizure_count}")
    print(f"  Predicted non-seizure segments:      {non_seizure_count}")
    print(f"  Min VQC seizure probability:         {probabilities.min():.6f}")
    print(f"  Max VQC seizure probability:         {probabilities.max():.6f}")
    print(f"  Mean VQC seizure probability:        {probabilities.mean():.6f}")
    print(f"  Recording-level seizure probability: {probabilities.mean():.6f}")
    print(f"    (aggregation: arithmetic mean of all segment probabilities)")
    print(f"  Binary decision threshold:           {service.threshold}")
    print(f"  Total inference time:                {total_time:.2f} s")
    print()
    print(f"  First 10 VQC seizure probabilities:")
    print(f"    {np.array2string(probabilities[:10], precision=6, separator=', ')}")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
