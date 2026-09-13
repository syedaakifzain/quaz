"""EDF file validation — channels, sampling rate, duration."""

from __future__ import annotations

import logging

import mne

from core.config import FeatureConfig
from core.exceptions import (
    EdfChannelError,
    EdfDurationError,
    EdfSamplingRateError,
)

logger = logging.getLogger(__name__)


def validate_channels(raw: mne.io.Raw, config: FeatureConfig) -> None:
    """Validate that all required channels exist in the recording.

    Channel names are normalized (stripped, uppercased) before comparison.

    Args:
        raw: MNE Raw object.
        config: Feature configuration with required channel list.

    Raises:
        EdfChannelError: If any required channels are missing.
    """
    available = {ch.strip().upper(): ch for ch in raw.ch_names}
    required = [ch.strip().upper() for ch in config.channels]

    missing = [ch for ch in required if ch not in available]

    if missing:
        # Map back to original case for the error message
        original_required = config.channels
        original_available = list(raw.ch_names)
        missing_original = [
            original_required[i]
            for i, ch in enumerate(required)
            if ch not in available
        ]
        raise EdfChannelError(
            missing_channels=missing_original,
            available_channels=original_available,
        )

    logger.debug("All %d required channels found.", len(required))


def validate_sampling_rate(raw: mne.io.Raw, expected_sfreq: float | None = None) -> None:
    """Validate the sampling rate is usable.

    Args:
        raw: MNE Raw object.
        expected_sfreq: If provided, verify the sampling rate matches.

    Raises:
        EdfSamplingRateError: If the sampling rate is invalid or mismatched.
    """
    sfreq = raw.info["sfreq"]

    if sfreq <= 0:
        raise EdfSamplingRateError(f"Invalid sampling rate: {sfreq} Hz")

    if expected_sfreq is not None and sfreq != expected_sfreq:
        raise EdfSamplingRateError(
            f"Sampling rate mismatch: expected {expected_sfreq} Hz, got {sfreq} Hz"
        )

    logger.debug("Sampling rate: %.1f Hz", sfreq)


def validate_duration(raw: mne.io.Raw, min_duration_seconds: float = 2.0) -> None:
    """Validate the recording duration is sufficient.

    Args:
        raw: MNE Raw object.
        min_duration_seconds: Minimum required duration.

    Raises:
        EdfDurationError: If the recording is too short.
    """
    duration = raw.times[-1] if len(raw.times) > 0 else 0.0

    if duration < min_duration_seconds:
        raise EdfDurationError(
            f"Recording duration ({duration:.1f}s) is less than the minimum "
            f"required ({min_duration_seconds:.1f}s)"
        )

    logger.debug("Duration: %.1f seconds", duration)


def validate_edf(raw: mne.io.Raw, config: FeatureConfig) -> None:
    """Run all validation checks on an EDF recording.

    Args:
        raw: MNE Raw object.
        config: Feature configuration.

    Raises:
        EdfChannelError, EdfSamplingRateError, EdfDurationError
    """
    validate_channels(raw, config)
    validate_sampling_rate(raw)
    validate_duration(raw, min_duration_seconds=config.segment_duration_seconds)
    logger.info("EDF validation passed.")
