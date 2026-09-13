"""CHB-MIT seizure annotation parser.

Parses the *-summary.txt files shipped with the CHB-MIT dataset to
extract per-file seizure intervals (start/end in seconds).

The .seizures binary sidecar files are NOT relied upon because their
format is not consistently documented. The summary text files are the
authoritative annotation source.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from core.exceptions import AnnotationParseError

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────


@dataclass
class SeizureInterval:
    """A single seizure interval within a recording."""
    start_seconds: float
    end_seconds: float


@dataclass
class FileAnnotation:
    """Seizure annotations for one EDF file."""
    file_name: str
    num_seizures: int
    seizure_intervals: list[SeizureInterval] = field(default_factory=list)


@dataclass
class SubjectAnnotations:
    """All annotations for one CHB-MIT subject."""
    subject_id: str
    sampling_rate_hz: float | None
    channel_names: list[str]
    file_annotations: dict[str, FileAnnotation] = field(default_factory=dict)


# ── Parsing ───────────────────────────────────────────────────


def parse_summary_file(summary_path: Path) -> SubjectAnnotations:
    """Parse a CHB-MIT summary text file.

    Args:
        summary_path: Path to a ``chbXX-summary.txt`` file.

    Returns:
        SubjectAnnotations with seizure intervals for each EDF listed.

    Raises:
        AnnotationParseError: If the file cannot be parsed.
    """
    if not summary_path.exists():
        raise AnnotationParseError(f"Summary file not found: {summary_path}")

    subject_id = summary_path.stem.split("-")[0]  # chb01-summary → chb01

    try:
        text = summary_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise AnnotationParseError(
            f"Failed to read summary file {summary_path}: {e}"
        ) from e

    # Extract sampling rate
    sampling_rate: float | None = None
    rate_match = re.search(r"Data Sampling Rate:\s*(\d+)\s*Hz", text)
    if rate_match:
        sampling_rate = float(rate_match.group(1))

    # Extract channel list
    channel_names: list[str] = []
    for ch_match in re.finditer(r"Channel\s+\d+:\s*(\S+)", text):
        ch_name = ch_match.group(1).strip()
        if ch_name not in channel_names:
            channel_names.append(ch_name)

    # Split into per-file blocks
    file_annotations: dict[str, FileAnnotation] = {}
    # Each file block starts with "File Name:"
    file_blocks = re.split(r"(?=File Name:)", text)

    for block in file_blocks:
        block = block.strip()
        if not block.startswith("File Name:"):
            continue

        fname_match = re.search(r"File Name:\s*(\S+)", block)
        if not fname_match:
            continue
        file_name = fname_match.group(1).strip()

        n_seizures_match = re.search(
            r"Number of Seizures in File:\s*(\d+)", block
        )
        num_seizures = int(n_seizures_match.group(1)) if n_seizures_match else 0

        intervals: list[SeizureInterval] = []
        # Find all seizure start/end pairs
        # The pattern handles both "Seizure Start Time:" and
        # "Seizure N Start Time:" variants used in some subjects.
        starts = re.findall(
            r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s*(\d+)\s*seconds", block
        )
        ends = re.findall(
            r"Seizure(?:\s+\d+)?\s+End\s+Time:\s*(\d+)\s*seconds", block
        )

        if len(starts) != len(ends):
            logger.warning(
                "Mismatched seizure start/end counts in %s for %s "
                "(starts=%d, ends=%d). Skipping mismatched pairs.",
                summary_path.name, file_name, len(starts), len(ends),
            )

        for s, e in zip(starts, ends):
            intervals.append(SeizureInterval(
                start_seconds=float(s),
                end_seconds=float(e),
            ))

        if len(intervals) != num_seizures:
            logger.warning(
                "Expected %d seizures for %s but parsed %d intervals.",
                num_seizures, file_name, len(intervals),
            )

        file_annotations[file_name] = FileAnnotation(
            file_name=file_name,
            num_seizures=num_seizures,
            seizure_intervals=intervals,
        )

    logger.info(
        "Parsed annotations for subject %s: %d files, %d with seizures",
        subject_id,
        len(file_annotations),
        sum(1 for fa in file_annotations.values() if fa.num_seizures > 0),
    )

    return SubjectAnnotations(
        subject_id=subject_id,
        sampling_rate_hz=sampling_rate,
        channel_names=channel_names,
        file_annotations=file_annotations,
    )


def get_seizure_intervals(
    subject_annotations: SubjectAnnotations,
    edf_filename: str,
) -> list[SeizureInterval]:
    """Get seizure intervals for a specific EDF file.

    Args:
        subject_annotations: Parsed annotations for the subject.
        edf_filename: Basename of the EDF file (e.g. ``chb01_03.edf``).

    Returns:
        List of SeizureInterval. Empty if no seizures or file not annotated.
    """
    fa = subject_annotations.file_annotations.get(edf_filename)
    if fa is None:
        logger.debug(
            "No annotation entry found for %s in subject %s — treating as non-seizure.",
            edf_filename, subject_annotations.subject_id,
        )
        return []
    return fa.seizure_intervals


def discover_summary_file(subject_dir: Path) -> Path | None:
    """Find the summary file for a CHB-MIT subject directory.

    Args:
        subject_dir: Path to a subject folder (e.g. ``data/training/chb01/``).

    Returns:
        Path to the summary file, or None if not found.
    """
    candidates = list(subject_dir.glob("*-summary.txt"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logger.warning(
            "Multiple summary files in %s: %s — using first.",
            subject_dir, [c.name for c in candidates],
        )
        return candidates[0]
    return None
