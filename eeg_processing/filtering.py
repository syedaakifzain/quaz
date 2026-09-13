"""Bandpass filtering — shared between training and inference."""

from __future__ import annotations

import logging

import mne

logger = logging.getLogger(__name__)


def bandpass_filter(
    raw: mne.io.Raw,
    low_hz: float,
    high_hz: float,
) -> mne.io.Raw:
    """Apply a bandpass FIR filter to an MNE Raw object.

    The filter is applied in-place on a *copy* of the Raw object so the
    caller's original data is not modified.

    Args:
        raw: MNE Raw object (will be copied).
        low_hz: Low cutoff frequency in Hz.
        high_hz: High cutoff frequency in Hz.

    Returns:
        Filtered Raw object (copy).
    """
    raw_filtered = raw.copy().load_data()
    raw_filtered.filter(
        l_freq=low_hz,
        h_freq=high_hz,
        method="fir",
        verbose=False,
    )
    logger.info(
        "Applied bandpass filter: %.1f–%.1f Hz", low_hz, high_hz,
    )
    return raw_filtered
