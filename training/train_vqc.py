"""VQC training orchestration — wraps quantum classifier for the training pipeline."""

from __future__ import annotations

import logging

import numpy as np

from core.config import VQCConfig
from quantum.classifier import build_vqc, train_vqc as _train, predict_vqc

logger = logging.getLogger(__name__)


def create_and_train_vqc(
    config: VQCConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_qubits: int = 4,
):
    """Build and train a VQC.

    Args:
        config: VQC hyperparameters.
        X_train: Scaled PCA training features (n, num_qubits).
        y_train: Training labels (n,).
        num_qubits: Number of qubits (must match PCA output dimension).

    Returns:
        Trained VQC instance.
    """
    vqc = build_vqc(config, num_qubits=num_qubits)
    vqc = _train(vqc, X_train, y_train)
    return vqc

