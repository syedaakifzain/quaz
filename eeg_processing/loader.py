"""EDF file loader using MNE with memory-efficient lazy loading."""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np

from core.exceptions import EdfFileNotFoundError, EdfInvalidFormatError
from core.models import EEGMetadata

logger = logging.getLogger(__name__)


def load_edf(file_path: str | Path, preload: bool = False) -> mne.io.Raw:
    """Load an EDF file using MNE.

    Args:
        file_path: Path to the EDF file.
        preload: If True, load data into memory immediately.
                 If False (default), only read the header for memory efficiency.

    Returns:
        MNE Raw object.

    Raises:
        EdfFileNotFoundError: If the file does not exist.
        EdfInvalidFormatError: If the file is not a valid EDF.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise EdfFileNotFoundError(f"EDF file not found: {file_path}")

    if file_path.suffix.lower() != ".edf":
        raise EdfInvalidFormatError(
            f"Expected .edf file extension, got '{file_path.suffix}': {file_path}"
        )

    try:
        raw = mne.io.read_raw_edf(str(file_path), preload=preload, verbose=False)
    except Exception as e:
        raise EdfInvalidFormatError(
            f"Failed to read EDF file '{file_path.name}': {e}"
        ) from e

    logger.info(
        "Loaded EDF: %s | %.1f Hz | %.1f s | %d channels",
        file_path.name,
        raw.info["sfreq"],
        raw.times[-1] if len(raw.times) > 0 else 0.0,
        len(raw.ch_names),
    )

    return raw


def extract_metadata(raw: mne.io.Raw, file_path: str | Path) -> EEGMetadata:
    """Extract metadata from an MNE Raw object.

    Args:
        raw: MNE Raw object (does not need to be preloaded).
        file_path: Original file path for provenance.

    Returns:
        EEGMetadata dataclass.
    """
    file_path = Path(file_path)

    return EEGMetadata(
        file_name=file_path.name,
        file_path=str(file_path),
        sampling_frequency=float(raw.info["sfreq"]),
        duration_seconds=float(raw.times[-1]) if len(raw.times) > 0 else 0.0,
        channel_count=len(raw.ch_names),
        channel_names=list(raw.ch_names),
        n_samples=raw.n_times,
    )
