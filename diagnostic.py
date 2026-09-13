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
print("=== SEIZURE WINDOWS ===")

for j, (a, b) in enumerate(intervals, 1):
    inside = (starts >= a) & (starts < b)
    around = (starts >= a - 30) & (starts < b + 30)

    print()
    print(f"Seizure {j}: {a}-{b}s")
    print(f"  Max probability in +/-30s: {p[around].max():.6f}")
    print(f"  Max probability inside:     {p[inside].max():.6f}")
    print(f"  Mean probability inside:    {p[inside].mean():.6f}")
    print(f"  >= 0.60 inside:             {int(np.sum((p >= 0.60) & inside))}")

    top = sorted(
        [
            (float(p[i]), float(starts[i]))
            for i in range(len(p))
            if around[i]
        ],
        reverse=True,
    )[:15]

    print("  Top probabilities around seizure:")
    for probability, start in top:
        print(f"    {start:8.1f}s -> {probability:.6f}")

print()
print("=== DONE ===")
