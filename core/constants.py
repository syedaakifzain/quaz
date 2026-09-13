"""Constants and enumerations for the Quantum EEG Signal Classifier."""

from enum import IntEnum


class Label(IntEnum):
    """Segment classification labels."""
    NON_SEIZURE = 0
    SEIZURE = 1


# Feature names in deterministic order (frozen)
FEATURE_NAMES: list[str] = [
    "wavelet_approx_energy",
    "wavelet_energy",
    "shannon_entropy",
    "delta_power",
    "hjorth_mobility",
]

# Required model artifact filenames
MODEL_ARTIFACT_FILES: list[str] = [
    "vqc_model.dill",
    "pre_scaler.joblib",
    "pca.joblib",
    "scaler.joblib",
    "feature_config.json",
    "preprocessing_config.json",
    "model_metadata.json",
]

# Medical disclaimer
MEDICAL_DISCLAIMER: str = (
    "This system is a research and educational prototype for EEG seizure activity "
    "classification. Model predictions are not a medical diagnosis and must not "
    "replace evaluation by a qualified healthcare professional."
)
