"""Quantum model — Variational Quantum Classifier with dynamic qubit count.

Phase 10: Implements the VQC using Qiskit 2.x and Qiskit Machine Learning 0.9.

Architecture:
    N PCA features → N qubits (default 4)
    ZZFeatureMap → RealAmplitudes → COBYLA → VQC (binary classification)

Uses qiskit_machine_learning.algorithms.VQC — the supported high-level
API in Qiskit ML 0.9.
"""

from __future__ import annotations

import logging

import numpy as np
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

from core.config import VQCConfig

logger = logging.getLogger(__name__)

# Default number of qubits (used when not overridden)
DEFAULT_NUM_QUBITS = 4


def build_vqc(config: VQCConfig, num_qubits: int = DEFAULT_NUM_QUBITS) -> VQC:
    """Construct a VQC with the specified number of qubits.

    Args:
        config: VQC hyperparameters.
        num_qubits: Number of qubits (must match PCA output dimension).

    Returns:
        Untrained VQC instance.
    """
    # Feature map: ZZFeatureMap
    feature_map = ZZFeatureMap(
        feature_dimension=num_qubits,
        reps=config.feature_map_reps,
        entanglement=config.entanglement,
    )

    # Variational ansatz: RealAmplitudes
    ansatz = RealAmplitudes(
        num_qubits=num_qubits,
        reps=config.ansatz_reps,
        entanglement=config.entanglement,
    )

    # Optimizer
    optimizer = COBYLA(maxiter=config.maxiter)

    vqc = VQC(
        num_qubits=num_qubits,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
    )

    logger.info(
        "Built VQC: %d qubits, feature_map=%s (reps=%d), "
        "ansatz=%s (reps=%d), optimizer=%s (maxiter=%d)",
        num_qubits, config.feature_map, config.feature_map_reps,
        config.ansatz, config.ansatz_reps,
        config.optimizer, config.maxiter,
    )

    return vqc


def train_vqc(
    vqc: VQC,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> VQC:
    """Train the VQC on the prepared training data.

    Args:
        vqc: Untrained VQC instance.
        X_train: Scaled PCA training features (n, 4).
        y_train: Training labels (n,).

    Returns:
        Trained VQC instance.
    """
    logger.info(
        "Starting VQC training: %d samples, %d features",
        X_train.shape[0], X_train.shape[1],
    )

    vqc.fit(X_train, y_train)

    logger.info("VQC training complete.")
    return vqc


def predict_proba_vqc(
    vqc: VQC,
    X: np.ndarray,
) -> np.ndarray:
    """Return positive-class (seizure) probability for each sample.

    Args:
        vqc: Trained VQC.
        X: Scaled PCA features (n, num_qubits).

    Returns:
        1-D array of shape (n_samples,) with seizure probabilities.
    """
    proba = vqc.predict_proba(X)

    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1]
    return proba.ravel()


def predict_vqc(
    vqc: VQC,
    X: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Generate predictions using the trained VQC with a custom threshold.

    Args:
        vqc: Trained VQC.
        X: Scaled PCA features (n, 4).
        threshold: Probability threshold for the positive class (seizure).

    Returns:
        Predicted labels array.
    """
    if threshold == 0.5:
        # Fast path if using standard threshold
        return vqc.predict(X)

    # Qiskit VQC predict_proba shape depends on how it is configured,
    # but normally it returns an array of shape (n_samples, n_classes).
    # Since it's binary classification, we check column 1.
    proba = vqc.predict_proba(X)
    
    # If the output is a 1D array of probabilities, just use that,
    # else if it's 2D (one-hot probability output), use column 1.
    if proba.ndim == 2 and proba.shape[1] >= 2:
        prob_positive = proba[:, 1]
    else:
        # Depending on Qiskit version, sometimes it's (n_samples, 1) or 1D
        prob_positive = proba.ravel()
        
    y_pred = (prob_positive >= threshold).astype(int)
    return y_pred
