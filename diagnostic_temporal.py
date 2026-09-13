from pathlib import Path
import numpy as np

from inference.pipeline import InferenceService
from eeg_processing.annotations import (
    discover_summary_file,
    parse_summary_file,
    get_seizure_intervals,
)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

EDF_PATH = Path("test_data/chb06_01.edf")
SUMMARY_PATH = Path("test_data/chb06-summary.txt")

MODEL_DIR = Path("models")
MODEL_VERSION = "v1.3.0"

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

service = InferenceService(
    artifacts_dir=MODEL_DIR,
    model_version=MODEL_VERSION,
)

print("\nRunning inference...")
result = service.predict(EDF_PATH)

probs = np.array([
    s.seizure_probability
    for s in result.segment_predictions
])

times = np.array([
    s.start_time_seconds
    for s in result.segment_predictions
])

# ------------------------------------------------------------
# GROUND TRUTH
# ------------------------------------------------------------

annotations = parse_summary_file(SUMMARY_PATH)
intervals = get_seizure_intervals(
    annotations,
    EDF_PATH.name,
)

y_true = np.zeros(len(times), dtype=int)

for i, t in enumerate(times):
    segment_end = t + 2.0

    for interval in intervals:
        if (
            t < interval.end_seconds
            and interval.start_seconds < segment_end
        ):
            y_true[i] = 1
            break

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return tp, fp, fn, precision, recall, f1


# ------------------------------------------------------------
# TEMPORAL RULES
# ------------------------------------------------------------

def temporal_predict(probs, threshold, window, required):
    """
    Predict seizure if at least `required` of the previous/current
    `window` segments have probability >= threshold.
    """

    binary = probs >= threshold
    predictions = np.zeros(len(probs), dtype=int)

    for i in range(len(probs)):
        start = max(0, i - window + 1)
        count = np.sum(binary[start:i + 1])

        if count >= required:
            predictions[i] = 1

    return predictions


# ------------------------------------------------------------
# EXPERIMENTS
# ------------------------------------------------------------

print("\n")
print("=" * 100)
print("TEMPORAL POST-PROCESSING EXPERIMENT")
print("=" * 100)

print(
    f"{'Rule':<15}"
    f"{'Threshold':<12}"
    f"{'TP':<8}"
    f"{'FP':<8}"
    f"{'FN':<8}"
    f"{'Precision':<14}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 100)

for threshold in [0.48, 0.50, 0.52, 0.55, 0.58, 0.60]:

    for window, required, name in [
        (3, 2, "2-of-3"),
        (5, 3, "3-of-5"),
    ]:

        y_pred = temporal_predict(
            probs,
            threshold,
            window,
            required,
        )

        tp, fp, fn, precision, recall, f1 = calculate_metrics(
            y_true,
            y_pred,
        )

        print(
            f"{name:<15}"
            f"{threshold:<12.2f}"
            f"{tp:<8}"
            f"{fp:<8}"
            f"{fn:<8}"
            f"{precision:<14.6f}"
            f"{recall:<12.6f}"
            f"{f1:<12.6f}"
        )

print("=" * 100)
print("DONE")
print("=" * 100)