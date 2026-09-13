"""Configuration loader — reads YAML configs into typed dataclasses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.exceptions import ConfigError, ConfigFileNotFoundError


# ── Typed Config Dataclasses ──────────────────────────────────


@dataclass(frozen=True)
class FilterConfig:
    """Bandpass filter parameters."""
    low_hz: float
    high_hz: float


@dataclass(frozen=True)
class WaveletConfig:
    """Wavelet decomposition parameters."""
    name: str
    level: int


@dataclass(frozen=True)
class DeltaBandConfig:
    """Delta frequency band boundaries."""
    low_hz: float
    high_hz: float


@dataclass(frozen=True)
class PCAConfig:
    """PCA dimensionality reduction parameters."""
    n_components: int


@dataclass(frozen=True)
class ScalerConfig:
    """Scaler / quantum input preparation parameters."""
    type: str
    feature_range_min: float
    feature_range_max: float

    @property
    def feature_range(self) -> tuple[float, float]:
        return (self.feature_range_min, self.feature_range_max)


@dataclass(frozen=True)
class FeatureConfig:
    """Complete feature extraction and preprocessing configuration."""
    channels: list[str]
    segment_duration_seconds: float
    filter: FilterConfig
    wavelet: WaveletConfig
    delta_band: DeltaBandConfig
    pca: PCAConfig
    scaler: ScalerConfig
    features: list[str]

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_features_per_channel(self) -> int:
        return len(self.features)

    @property
    def total_features(self) -> int:
        return self.n_channels * self.n_features_per_channel

    def schema_hash(self) -> str:
        """Deterministic hash of the feature schema for validation."""
        schema = {
            "channels": self.channels,
            "features": self.features,
            "segment_duration_seconds": self.segment_duration_seconds,
            "wavelet_name": self.wavelet.name,
            "wavelet_level": self.wavelet.level,
            "delta_low_hz": self.delta_band.low_hz,
            "delta_high_hz": self.delta_band.high_hz,
            "pca_n_components": self.pca.n_components,
        }
        schema_str = json.dumps(schema, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()


@dataclass(frozen=True)
class VQCConfig:
    """VQC hyperparameters."""
    feature_map: str
    feature_map_reps: int
    ansatz: str
    ansatz_reps: int
    entanglement: str
    optimizer: str
    maxiter: int
    random_seed: int


@dataclass(frozen=True)
class DatasetConfig:
    """Training dataset configuration."""
    base_dir: str
    subjects: list[str]


@dataclass(frozen=True)
class BalancingConfig:
    """Class balancing strategy."""
    strategy: str
    random_seed: int


@dataclass(frozen=True)
class TrainingOutputConfig:
    """Training output paths."""
    model_version: str
    artifacts_dir: str


@dataclass(frozen=True)
class TrainingConfig:
    """Complete training configuration."""
    dataset: DatasetConfig
    balancing: BalancingConfig
    evaluation_method: str
    vqc: VQCConfig
    output: TrainingOutputConfig


@dataclass(frozen=True)
class ModelConfig:
    """Model artifact location."""
    version: str
    artifacts_dir: str


@dataclass(frozen=True)
class UploadConfig:
    """Upload constraints."""
    max_file_size_mb: int
    allowed_extensions: list[str]


@dataclass(frozen=True)
class UIConfig:
    """UI display settings."""
    app_name: str
    app_subtitle: str
    page_icon: str
    theme: str


@dataclass(frozen=True)
class AggregationConfig:
    """Recording-level aggregation settings."""
    min_seizure_segments: int


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""
    model: ModelConfig
    upload: UploadConfig
    ui: UIConfig
    aggregation: AggregationConfig


# ── YAML Loading Functions ────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        raise ConfigFileNotFoundError(f"Configuration file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML file {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
    return data


def load_feature_config(path: Path | str | None = None) -> FeatureConfig:
    """Load feature configuration from YAML.

    Args:
        path: Path to features.yaml. Defaults to config/features.yaml
              relative to project root.
    """
    if path is None:
        path = _project_root() / "config" / "features.yaml"
    path = Path(path)
    data = _load_yaml(path)

    try:
        return FeatureConfig(
            channels=data["channels"],
            segment_duration_seconds=float(data["segment_duration_seconds"]),
            filter=FilterConfig(
                low_hz=float(data["filter"]["low_hz"]),
                high_hz=float(data["filter"]["high_hz"]),
            ),
            wavelet=WaveletConfig(
                name=data["wavelet"]["name"],
                level=int(data["wavelet"]["level"]),
            ),
            delta_band=DeltaBandConfig(
                low_hz=float(data["delta_band"]["low_hz"]),
                high_hz=float(data["delta_band"]["high_hz"]),
            ),
            pca=PCAConfig(
                n_components=int(data["pca"]["n_components"]),
            ),
            scaler=ScalerConfig(
                type=data["scaler"]["type"],
                feature_range_min=float(data["scaler"]["feature_range_min"]),
                feature_range_max=float(data["scaler"]["feature_range_max"]),
            ),
            features=data["features"],
        )
    except KeyError as e:
        raise ConfigError(f"Missing required key in {path}: {e}") from e


def load_training_config(path: Path | str | None = None) -> TrainingConfig:
    """Load training configuration from YAML.

    Args:
        path: Path to training.yaml. Defaults to config/training.yaml
              relative to project root.
    """
    if path is None:
        path = _project_root() / "config" / "training.yaml"
    path = Path(path)
    data = _load_yaml(path)

    try:
        return TrainingConfig(
            dataset=DatasetConfig(
                base_dir=data["dataset"]["base_dir"],
                subjects=data["dataset"]["subjects"],
            ),
            balancing=BalancingConfig(
                strategy=data["balancing"]["strategy"],
                random_seed=int(data["balancing"]["random_seed"]),
            ),
            evaluation_method=data["evaluation"]["method"],
            vqc=VQCConfig(
                feature_map=data["vqc"]["feature_map"],
                feature_map_reps=int(data["vqc"]["feature_map_reps"]),
                ansatz=data["vqc"]["ansatz"],
                ansatz_reps=int(data["vqc"]["ansatz_reps"]),
                entanglement=data["vqc"]["entanglement"],
                optimizer=data["vqc"]["optimizer"],
                maxiter=int(data["vqc"]["maxiter"]),
                random_seed=int(data["vqc"]["random_seed"]),
            ),
            output=TrainingOutputConfig(
                model_version=data["output"]["model_version"],
                artifacts_dir=data["output"]["artifacts_dir"],
            ),
        )
    except KeyError as e:
        raise ConfigError(f"Missing required key in {path}: {e}") from e


def load_app_config(path: Path | str | None = None) -> AppConfig:
    """Load application configuration from YAML.

    Args:
        path: Path to app.yaml. Defaults to config/app.yaml
              relative to project root.
    """
    if path is None:
        path = _project_root() / "config" / "app.yaml"
    path = Path(path)
    data = _load_yaml(path)

    try:
        return AppConfig(
            model=ModelConfig(
                version=data["model"]["version"],
                artifacts_dir=data["model"]["artifacts_dir"],
            ),
            upload=UploadConfig(
                max_file_size_mb=int(data["upload"]["max_file_size_mb"]),
                allowed_extensions=data["upload"]["allowed_extensions"],
            ),
            ui=UIConfig(
                app_name=data["ui"]["app_name"],
                app_subtitle=data["ui"]["app_subtitle"],
                page_icon=data["ui"]["page_icon"],
                theme=data["ui"]["theme"],
            ),
            aggregation=AggregationConfig(
                min_seizure_segments=int(data["aggregation"]["min_seizure_segments"]),
            ),
        )
    except KeyError as e:
        raise ConfigError(f"Missing required key in {path}: {e}") from e


def _project_root() -> Path:
    """Resolve the project root directory (parent of core/)."""
    return Path(__file__).resolve().parent.parent
