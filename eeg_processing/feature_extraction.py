"""20-dimensional EEG feature extraction — shared between training and inference.

For every configured channel the extractor computes five features:

1. **Wavelet Approx Energy** — sum of squared approximation coefficients
   at the deepest decomposition level (0–8 Hz delta/theta at fs=256 Hz).
2. **Wavelet Energy** — sum of squared detail coefficients at the
   deepest decomposition level (8–16 Hz alpha/low-beta at fs=256 Hz).
3. **Shannon Entropy** — entropy of the normalized power distribution
   of the segment signal.
4. **Delta Band Power** — relative power in the configured delta
   frequency band (via Welch PSD).
5. **Hjorth Mobility** — ratio of the standard deviation of the first
   derivative to the standard deviation of the signal.

With 4 channels × 5 features the output is always a 20-element vector.

The feature order is deterministic:
    [ch0_approx, ch0_wavelet, ch0_entropy, ch0_delta, ch0_hjorth,
     ch1_approx, ch1_wavelet, ch1_entropy, ch1_delta, ch1_hjorth,
     ...
    ]
"""

from __future__ import annotations

import logging

import numpy as np
import pywt
from scipy import signal as sp_signal

from core.config import FeatureConfig
from core.exceptions import FeatureExtractionError

logger = logging.getLogger(__name__)

# Tiny constant to avoid log(0)
_EPS = 1e-12

# Version-safe trapezoidal integration (NumPy 2.x renamed trapz → trapezoid)
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ── Individual Feature Functions ──────────────────────────────


def _wavelet_approx_energy(
    segment: np.ndarray,
    wavelet_name: str,
    level: int,
) -> float:
    """Sum of squared approximation coefficients (0–8 Hz at fs=256 Hz).

    coeffs[0] from pywt.wavedec is the approximation at the deepest
    decomposition level — for db4, level=4, fs=256 Hz this is the
    0–8 Hz (delta/theta) band.  Empirically this is the single
    strongest seizure discriminator (Cohen's d = 0.974).
    """
    coeffs = pywt.wavedec(segment, wavelet_name, level=level)
    approx = coeffs[0]  # approximation: lowest-frequency band
    return float(np.sum(approx ** 2))


def _wavelet_energy(
    segment: np.ndarray,
    wavelet_name: str,
    level: int,
) -> float:
    """Sum of squared detail coefficients at the deepest decomposition level."""
    coeffs = pywt.wavedec(segment, wavelet_name, level=level)
    # coeffs[1] = level-4 detail (8–16 Hz alpha/low-beta band at fs=256 Hz)
    detail = coeffs[1]
    return float(np.sum(detail ** 2))


def _shannon_entropy(segment: np.ndarray) -> float:
    """Shannon entropy of the normalised amplitude distribution."""
    # Use histogram-based probability estimation
    hist, _ = np.histogram(segment, bins=64, density=True)
    hist = hist[hist > 0]  # drop zero bins
    # Normalise so it sums to 1 (density=True + bin width ≠ 1)
    p = hist / (hist.sum() + _EPS)
    return float(-np.sum(p * np.log2(p + _EPS)))


def _delta_band_power(
    segment: np.ndarray,
    sfreq: float,
    delta_low_hz: float,
    delta_high_hz: float,
) -> float:
    """Relative power in the delta frequency band via Welch PSD."""
    nperseg = min(len(segment), int(sfreq * 2))
    if nperseg < 4:
        return 0.0
    freqs, psd = sp_signal.welch(segment, fs=sfreq, nperseg=nperseg)
    # Total power
    total_power = _trapz(psd, freqs) + _EPS
    # Delta band mask
    delta_mask = (freqs >= delta_low_hz) & (freqs <= delta_high_hz)
    delta_power = _trapz(psd[delta_mask], freqs[delta_mask])
    return float(delta_power / total_power)


def _hjorth_mobility(segment: np.ndarray) -> float:
    """Hjorth mobility: std(dx) / std(x)."""
    dx = np.diff(segment)
    std_x = np.std(segment) + _EPS
    std_dx = np.std(dx)
    return float(std_dx / std_x)


# ── Public API ────────────────────────────────────────────────


def extract_features_for_segment(
    segment_data: np.ndarray,
    sfreq: float,
    config: FeatureConfig,
) -> np.ndarray:
    """Extract the 16-dimensional feature vector for one segment.

    Args:
        segment_data: Shape ``(n_channels, n_samples_per_segment)``.
        sfreq: Sampling frequency in Hz.
        config: Feature configuration.

    Returns:
        1-D numpy array of shape ``(total_features,)`` — typically 16.

    Raises:
        FeatureExtractionError: If extraction fails or dimension mismatch.
    """
    expected_dim = config.total_features  # 4 channels × 5 features = 20

    features: list[float] = []
    for ch_idx in range(config.n_channels):
        ch_signal = segment_data[ch_idx]

        try:
            features.append(_wavelet_approx_energy(
                ch_signal, config.wavelet.name, config.wavelet.level,
            ))
            features.append(_wavelet_energy(
                ch_signal, config.wavelet.name, config.wavelet.level,
            ))
            features.append(_shannon_entropy(ch_signal))
            features.append(_delta_band_power(
                ch_signal, sfreq,
                config.delta_band.low_hz, config.delta_band.high_hz,
            ))
            features.append(_hjorth_mobility(ch_signal))
        except Exception as e:
            raise FeatureExtractionError(
                f"Feature extraction failed for channel index {ch_idx}: {e}"
            ) from e

    feature_vec = np.array(features, dtype=np.float64)

    if feature_vec.shape[0] != expected_dim:
        raise FeatureExtractionError(
            f"Feature dimension mismatch: expected {expected_dim}, "
            f"got {feature_vec.shape[0]}"
        )

    # Replace any NaN/Inf with 0 and warn
    bad_mask = ~np.isfinite(feature_vec)
    if bad_mask.any():
        n_bad = int(bad_mask.sum())
        logger.warning(
            "Replaced %d non-finite feature values with 0.", n_bad,
        )
        feature_vec[bad_mask] = 0.0

    return feature_vec


def extract_features_batch(
    segments_data: list[np.ndarray],
    sfreq: float,
    config: FeatureConfig,
) -> np.ndarray:
    """Extract features for a batch of segments.

    Args:
        segments_data: List of arrays, each ``(n_channels, n_samples)``.
        sfreq: Sampling frequency in Hz.
        config: Feature configuration.

    Returns:
        2-D numpy array of shape ``(n_segments, total_features)``.
    """
    rows = [
        extract_features_for_segment(seg, sfreq, config)
        for seg in segments_data
    ]
    return np.vstack(rows)
