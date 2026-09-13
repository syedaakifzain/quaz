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

seizures = [
    (1724, 1738),
    (7461, 7476),
    (13525, 13540),
]

threshold = 0.60

print()
print("=== FALSE POSITIVE TIMING ANALYSIS ===")
print()

positive = p >= threshold

for window in [10, 20, 30, 60, 120, 300]:
    near_seizure = np.zeros(len(p), dtype=bool)

    for a, b in seizures:
        near_seizure |= (
            (starts >= a - window)
            & (starts <= b + window)
        )

    fp_near = np.sum(positive & near_seizure)
    fp_far = np.sum(positive & ~near_seizure)

    print(
        f"Within +/-{window:3d}s: "
        f"{fp_near:4d} positive segments | "
        f"Far from seizure: {fp_far:4d}"
    )

print()
print("=== POSITIVE SEGMENTS BY SEIZURE ===")

for j, (a, b) in enumerate(seizures, 1):

    inside = (
        (starts >= a)
        & (starts < b)
    )

    near = (
        (starts >= a - 60)
        & (starts < b + 60)
    )

    print()
    print(f"Seizure {j}: {a}-{b}s")

    print(
        "  >=0.60 inside: ",
        int(np.sum(positive & inside))
    )

    print(
        "  >=0.60 within +/-60s: ",
        int(np.sum(positive & near))
    )

    print(
        "  >=0.60 before seizure: ",
        int(np.sum(
            positive
            & (starts >= a - 60)
            & (starts < a)
        ))
    )

    print(
        "  >=0.60 after seizure:  ",
        int(np.sum(
            positive
            & (starts >= b)
            & (starts < b + 60)
        ))
    )

print()
print("=== DONE ===")