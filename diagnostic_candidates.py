import numpy as np
from pathlib import Path
from inference.pipeline import InferenceService


service = InferenceService(
    Path("models"),
    "v1.3.0",
)

result = service.predict(
    Path("test_data/chb06_01.edf")
)

p = np.array([
    x.seizure_probability
    for x in result.segment_predictions
])

starts = np.array([
    x.start_time_seconds
    for x in result.segment_predictions
])


# Ground-truth seizure intervals from chb06-summary.txt
intervals = [
    (1724, 1738),
    (7461, 7476),
    (13525, 13540),
]


print()
print("=" * 100)
print("CANDIDATE SEIZURE DETECTION ANALYSIS — V1.3.0")
print("=" * 100)

for j, (a, b) in enumerate(intervals, 1):

    inside = (starts >= a) & (starts < b)

    # Look 60 seconds before and after the annotated seizure
    around = (starts >= a - 60) & (starts < b + 60)

    print()
    print(f"SEIZURE {j}: {a}-{b}s")
    print("-" * 80)

    print(
        f"Max probability inside: "
        f"{p[inside].max():.6f}"
    )

    print(
        f"Mean probability inside: "
        f"{p[inside].mean():.6f}"
    )

    print(
        f"Max probability +/-60s: "
        f"{p[around].max():.6f}"
    )

    print(
        f">=0.60 inside: "
        f"{int(np.sum((p >= 0.60) & inside))}"
    )

    print(
        f">=0.65 inside: "
        f"{int(np.sum((p >= 0.65) & inside))}"
    )

    print(
        f">=0.70 inside: "
        f"{int(np.sum((p >= 0.70) & inside))}"
    )

    print()
    print("Top 10 probabilities around seizure:")

    candidates = [
        (float(p[i]), float(starts[i]))
        for i in range(len(p))
        if around[i]
    ]

    candidates.sort(reverse=True)

    for probability, start in candidates[:10]:

        if a <= start < b:
            location = "INSIDE"
        elif start < a:
            location = "BEFORE"
        else:
            location = "AFTER"

        print(
            f"  {start:8.1f}s -> "
            f"{probability:.6f}  {location}"
        )


print()
print("=" * 100)
print("GLOBAL PROBABILITY DISTRIBUTION")
print("=" * 100)

print(f"Total segments: {len(p)}")
print(f"Maximum probability: {p.max():.6f}")
print(f"Mean probability: {p.mean():.6f}")

for threshold in [0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80]:

    print(
        f">={threshold:.2f}: "
        f"{int(np.sum(p >= threshold))} segments"
    )


print()
print("=" * 100)
print("DONE")
print("=" * 100)