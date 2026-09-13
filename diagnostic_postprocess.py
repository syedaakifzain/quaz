import numpy as np
from pathlib import Path
from inference.pipeline import InferenceService

service = InferenceService(Path("models"), "v1.3.0")
result = service.predict(Path("test_data/chb06_01.edf"))

p = np.array([
    x.seizure_probability
    for x in result.segment_predictions
])

starts = np.array([
    x.start_time_seconds
    for x in result.segment_predictions
])

# Ground truth seizure intervals
intervals = [
    (1724, 1738),
    (7461, 7476),
    (13525, 13540),
]

# ---------------------------------------------------------
# Ground truth: ANY overlap with a 2-second segment
# ---------------------------------------------------------

y_true = np.zeros(len(p), dtype=int)

for i, start in enumerate(starts):
    end = start + 2.0

    for a, b in intervals:
        if start < b and end > a:
            y_true[i] = 1
            break


def metrics(name, y_pred):

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / len(y_true)

    print(
        f"{name:<25}"
        f" TP={tp:<4}"
        f" FP={fp:<4}"
        f" FN={fn:<4}"
        f" Precision={precision:.6f}"
        f" Recall={recall:.6f}"
        f" F1={f1:.6f}"
        f" Spec={specificity:.6f}"
        f" Acc={accuracy:.6f}"
    )


# ---------------------------------------------------------
# Strategy A: current threshold
# ---------------------------------------------------------

threshold = 0.60

current = (p >= threshold).astype(int)

print()
print("=" * 120)
print("POST-PROCESSING EXPERIMENTS — v1.3.0")
print("=" * 120)
print()

metrics("A: threshold 0.60", current)


# ---------------------------------------------------------
# Strategy B: 2-of-3
# ---------------------------------------------------------

binary = p >= threshold

two_of_three = np.zeros(len(p), dtype=int)

for i in range(len(p)):

    left = max(0, i - 1)
    right = min(len(p), i + 2)

    if np.sum(binary[left:right]) >= 2:
        two_of_three[i] = 1

metrics("B: 2-of-3", two_of_three)


# ---------------------------------------------------------
# Strategy C: 3-of-5
# ---------------------------------------------------------

three_of_five = np.zeros(len(p), dtype=int)

for i in range(len(p)):

    left = max(0, i - 2)
    right = min(len(p), i + 3)

    if np.sum(binary[left:right]) >= 3:
        three_of_five[i] = 1

metrics("C: 3-of-5", three_of_five)


# ---------------------------------------------------------
# Strategy D: moving average
# ---------------------------------------------------------

def moving_average(values, radius):

    result = np.zeros(len(values))

    for i in range(len(values)):

        left = max(0, i - radius)
        right = min(len(values), i + radius + 1)

        result[i] = np.mean(values[left:right])

    return result


for radius in [1, 2, 3]:

    smoothed = moving_average(p, radius)

    for threshold in [0.55, 0.58, 0.60, 0.62]:

        prediction = (smoothed >= threshold).astype(int)

        metrics(
            f"D: avg r={radius}, t={threshold:.2f}",
            prediction
        )


print()
print("=" * 120)
print("DONE")
print("=" * 120)