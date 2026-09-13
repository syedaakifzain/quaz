import numpy as np
from pathlib import Path
from inference.pipeline import InferenceService

service = InferenceService(Path("models"), "v1.3.0")
result = service.predict(Path("test_data/chb06_01.edf"))

p = np.array([x.seizure_probability for x in result.segment_predictions])
starts = np.array([x.start_time_seconds for x in result.segment_predictions])

intervals = [
    (1724, 1738),
    (7461, 7476),
    (13525, 13540),
]

print()
print("=" * 80)
print("CURRENT V1.3.0 DETECTIONS VS GROUND TRUTH")
print("=" * 80)

for j, (a, b) in enumerate(intervals, 1):

    inside = (starts >= a) & (starts < b)

    print()
    print(f"SEIZURE {j}: {a}-{b}s")
    print("-" * 50)

    print("Inside seizure:")
    for i in np.where(inside)[0]:
        print(
            f"  {starts[i]:8.1f}s -> "
            f"{p[i]:.6f} "
            f"{'DETECTED' if p[i] >= 0.60 else 'missed'}"
        )

    print()
    print("Top 10 probabilities in +/- 30s:")

    around = (starts >= a - 30) & (starts < b + 30)

    indices = np.where(around)[0]
    indices = sorted(indices, key=lambda i: p[i], reverse=True)[:10]

    for i in indices:
        location = (
            "INSIDE"
            if a <= starts[i] < b
            else "BEFORE"
            if starts[i] < a
            else "AFTER"
        )

        print(
            f"  {starts[i]:8.1f}s -> "
            f"{p[i]:.6f}  {location}"
        )

print()
print("=" * 80)
print("ALL CURRENT TRUE POSITIVES")
print("=" * 80)

for i in np.where(p >= 0.60)[0]:

    matched = False

    for a, b in intervals:
        if a <= starts[i] < b:
            matched = True

    if matched:
        print(
            f"  {starts[i]:8.1f}s -> {p[i]:.6f}  TRUE POSITIVE"
        )

print()
print("=" * 80)
print("DONE")
print("=" * 80)
