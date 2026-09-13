"""Shared EEG processing pipeline — used by BOTH training and inference.

This is the single source of truth for:
  Load EDF → Validate → Channel Selection → Bandpass Filter → Segmentation

DO NOT create a separate pipeline for inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import mne

from core.config import FeatureConfig
from eeg_processing.loader import load_edf, extract_metadata
from eeg_processing.validator import validate_edf
from eeg_processing.channel_selector import select_channels
from eeg_processing.filtering import bandpass_filter
from eeg_processing.segmentation import segment_recording, EEGSegment

logger = logging.getLogger(__name__)


def preprocess_edf(
    file_path: str | Path,
    config: FeatureConfig,
) -> tuple[list[EEGSegment], float]:
    """Run the full shared preprocessing pipeline on an EDF file.

    Steps:
        1. Load EDF
        2. Validate channels / sampling rate / duration
        3. Select configured channels
        4. Apply bandpass filter
        5. Segment into non-overlapping windows

    Args:
        file_path: Path to the EDF file.
        config: Feature configuration (channels, filter, segment duration).

    Returns:
        Tuple of (list of EEGSegment, sampling_frequency).
    """
    file_path = Path(file_path)

    # 1. Load
    raw = load_edf(file_path, preload=False)

    # 2. Validate
    validate_edf(raw, config)

    sfreq = raw.info["sfreq"]

    # 3. Channel selection
    raw = select_channels(raw, config.channels)

    # 4. Bandpass filter (loads data into memory)
    raw = bandpass_filter(raw, config.filter.low_hz, config.filter.high_hz)

    # 5. Get numpy data (channels × samples)
    data = raw.get_data()  # shape: (n_channels, n_samples)

    # 6. Segment
    segments = segment_recording(data, sfreq, config.segment_duration_seconds)

    logger.info(
        "Preprocessed %s -> %d segments (%.1f Hz, %.1fs each)",
        file_path.name, len(segments), sfreq, config.segment_duration_seconds,
    )

    return segments, sfreq
