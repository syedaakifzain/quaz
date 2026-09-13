"""Channel selection — pick the configured EEG channels from a Raw object."""

from __future__ import annotations

import logging
from typing import Sequence

import mne

logger = logging.getLogger(__name__)


def select_channels(
    raw: mne.io.Raw,
    required_channels: Sequence[str],
) -> mne.io.Raw:
    """Select and reorder channels from an MNE Raw object.

    Channel matching is case-insensitive.  The returned Raw object
    contains only the requested channels in the specified order.

    Args:
        raw: MNE Raw object (original is not modified).
        required_channels: Ordered list of channel names to keep.

    Returns:
        New Raw object with only the required channels.

    Raises:
        ValueError: If any required channel is not present.
    """
    available = {ch.strip().upper(): ch for ch in raw.ch_names}
    pick_names: list[str] = []

    for req in required_channels:
        key = req.strip().upper()
        if key not in available:
            raise ValueError(
                f"Required channel '{req}' not found. "
                f"Available: {list(raw.ch_names)}"
            )
        pick_names.append(available[key])

    raw_selected = raw.copy().pick(pick_names)
    logger.info(
        "Selected %d channels: %s", len(pick_names), pick_names,
    )
    return raw_selected
