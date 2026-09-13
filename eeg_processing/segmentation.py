"""Segmentation — split a continuous EEG recording into fixed-duration windows.

Shared between training and inference.  The segment duration and the
recording's sampling frequency together determine the number of samples
per window; the code never hard-codes a sample count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EEGSegment:
    """One fixed-duration EEG segment."""
    index: int
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    data: np.ndarray  # shape: (n_channels, n_samples_per_segment)


def segment_recording(
    data: np.ndarray,
    sfreq: float,
    segment_duration_seconds: float,
) -> list[EEGSegment]:
    """Segment a multi-channel EEG recording into non-overlapping windows.

    Args:
        data: EEG data array of shape ``(n_channels, n_total_samples)``.
        sfreq: Sampling frequency in Hz.
        segment_duration_seconds: Duration of each segment in seconds.

    Returns:
        List of ``EEGSegment`` objects.  Incomplete trailing windows are
        discarded (floor division).
    """
    n_channels, n_total_samples = data.shape
    samples_per_segment = int(sfreq * segment_duration_seconds)

    if samples_per_segment <= 0:
        raise ValueError(
            f"Invalid segment size: sfreq={sfreq}, "
            f"duration={segment_duration_seconds}s -> {samples_per_segment} samples"
        )

    n_segments = n_total_samples // samples_per_segment

    segments: list[EEGSegment] = []
    for i in range(n_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segments.append(EEGSegment(
            index=i,
            start_sample=start,
            end_sample=end,
            start_seconds=start / sfreq,
            end_seconds=end / sfreq,
            data=data[:, start:end],
        ))

    logger.info(
        "Segmented %d samples -> %d segments of %.1fs (%d samples each). "
        "%d trailing samples discarded.",
        n_total_samples, n_segments, segment_duration_seconds,
        samples_per_segment,
        n_total_samples - n_segments * samples_per_segment,
    )

    return segments
