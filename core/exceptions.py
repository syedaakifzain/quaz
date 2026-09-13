"""Custom exception hierarchy for the Quantum EEG Signal Classifier."""


class QEEGError(Exception):
    """Base exception for all project errors."""


# ── EDF / File Errors ──────────────────────────────────────────

class EdfError(QEEGError):
    """Base exception for EDF-related errors."""


class EdfFileNotFoundError(EdfError):
    """Raised when the EDF file does not exist."""


class EdfInvalidFormatError(EdfError):
    """Raised when the file is not a valid EDF."""


class EdfChannelError(EdfError):
    """Raised when required channels are missing from the EDF."""

    def __init__(self, missing_channels: list[str], available_channels: list[str]):
        self.missing_channels = missing_channels
        self.available_channels = available_channels
        msg = (
            f"Missing required channels: {missing_channels}. "
            f"Available channels: {available_channels}"
        )
        super().__init__(msg)


class EdfSamplingRateError(EdfError):
    """Raised when the sampling rate is unsupported or inconsistent."""


class EdfDurationError(EdfError):
    """Raised when the recording duration is too short."""


# ── Model / Artifact Errors ───────────────────────────────────

class ModelError(QEEGError):
    """Base exception for model-related errors."""


class ModelArtifactNotFoundError(ModelError):
    """Raised when required model artifacts are missing."""

    def __init__(self, missing_files: list[str], artifacts_dir: str):
        self.missing_files = missing_files
        self.artifacts_dir = artifacts_dir
        msg = (
            f"Missing model artifacts in '{artifacts_dir}': {missing_files}. "
            f"Run the training pipeline first: python scripts/train.py"
        )
        super().__init__(msg)


class ModelVersionMismatchError(ModelError):
    """Raised when model metadata does not match expected configuration."""


class ModelSchemaError(ModelError):
    """Raised when feature schema or preprocessing config does not match."""


# ── Feature Extraction Errors ─────────────────────────────────

class FeatureExtractionError(QEEGError):
    """Raised when feature extraction fails for a segment."""


# ── Configuration Errors ──────────────────────────────────────

class ConfigError(QEEGError):
    """Raised when configuration is invalid or missing."""


class ConfigFileNotFoundError(ConfigError):
    """Raised when a configuration file does not exist."""


# ── Training Errors ───────────────────────────────────────────

class TrainingError(QEEGError):
    """Base exception for training pipeline errors."""


class AnnotationParseError(TrainingError):
    """Raised when seizure annotation parsing fails."""


class DatasetBuildError(TrainingError):
    """Raised when dataset construction fails."""


# ── Inference Errors ──────────────────────────────────────────

class InferenceError(QEEGError):
    """Base exception for inference pipeline errors."""
