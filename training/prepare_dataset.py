"""Dataset preparation — subject-aware splitting, balancing, PCA, and scaling.

Phase 6: Leave-One-Subject-Out (LOSO) splitting.
Phase 7: Training-only class balancing via deterministic undersampling.
Phase 8: PCA (16 → 4 dimensions).
Phase 9: MinMaxScaler to [0, π].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from core.config import FeatureConfig, BalancingConfig
from core.constants import Label
from training.build_dataset import SegmentMeta

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────


@dataclass
class LOSOFold:
    """One Leave-One-Subject-Out fold."""
    fold_index: int
    train_subjects: list[str]
    test_subject: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta_train: list[SegmentMeta]
    meta_test: list[SegmentMeta]


@dataclass
class PreparedFold:
    """A LOSO fold after balancing, pre-scaling, PCA, and scaling."""
    fold_index: int
    train_subjects: list[str]
    test_subject: str
    # After balancing
    X_train_balanced: np.ndarray
    y_train_balanced: np.ndarray
    train_segments_before_balance: int
    train_seizure_before_balance: int
    train_nonseizure_before_balance: int
    train_segments_after_balance: int
    # After preparation
    X_train_prepared: np.ndarray
    X_test_prepared: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    test_segments: int
    test_seizure: int
    test_nonseizure: int
    # Fitted objects
    pre_scaler: StandardScaler
    pca: PCA
    scaler: MinMaxScaler


# ── Subject-Aware LOSO Split ─────────────────────────────────


def create_loso_folds(
    X: np.ndarray,
    y: np.ndarray,
    metadata: list[SegmentMeta],
    subjects: list[str],
) -> list[LOSOFold]:
    """Create Leave-One-Subject-Out folds.

    Each fold holds one subject out as the test set while
    training on all remaining subjects.
    """
    subject_ids = np.array([m.subject_id for m in metadata])
    folds: list[LOSOFold] = []

    for fold_idx, test_subj in enumerate(sorted(subjects)):
        test_mask = subject_ids == test_subj
        train_mask = ~test_mask

        train_subjects = sorted(set(subject_ids[train_mask]))

        folds.append(LOSOFold(
            fold_index=fold_idx,
            train_subjects=train_subjects,
            test_subject=test_subj,
            X_train=X[train_mask],
            y_train=y[train_mask],
            X_test=X[test_mask],
            y_test=y[test_mask],
            meta_train=[m for m, keep in zip(metadata, train_mask) if keep],
            meta_test=[m for m, keep in zip(metadata, test_mask) if keep],
        ))

        logger.info(
            "LOSO Fold %d: test=%s, train subjects=%s, "
            "train=%d segments, test=%d segments",
            fold_idx, test_subj, train_subjects,
            int(train_mask.sum()), int(test_mask.sum()),
        )

    return folds


# ── Class Balancing ───────────────────────────────────────────


def balance_training_data(
    X: np.ndarray,
    y: np.ndarray,
    config: BalancingConfig,
    ratio: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Balance the training partition by undersampling the majority class.

    Args:
        X: Training feature matrix.
        y: Training labels.
        config: Balancing configuration (seed).
        ratio: Ratio of non-seizure to seizure samples (e.g., 3.0 means 3:1).

    Returns:
        (X_balanced, y_balanced)
    """
    rng = np.random.RandomState(config.random_seed)

    seizure_mask = y == Label.SEIZURE
    non_seizure_mask = y == Label.NON_SEIZURE

    n_seizure = int(seizure_mask.sum())
    n_non_seizure = int(non_seizure_mask.sum())

    if n_seizure == 0:
        logger.warning("No seizure samples in training data — returning as-is.")
        return X, y

    target_non_seizure = int(n_seizure * ratio)
    target_non_seizure = min(target_non_seizure, n_non_seizure)

    if n_non_seizure <= target_non_seizure:
        logger.info("Already balanced or near ratio (%d seizure, %d non-seizure).", n_seizure, n_non_seizure)
        return X, y

    non_seizure_indices = np.where(non_seizure_mask)[0]
    sampled_indices = rng.choice(non_seizure_indices, size=target_non_seizure, replace=False)

    seizure_indices = np.where(seizure_mask)[0]
    keep_indices = np.sort(np.concatenate([seizure_indices, sampled_indices]))

    X_balanced = X[keep_indices]
    y_balanced = y[keep_indices]

    logger.info(
        "Balanced training data: %d seizure + %d non-seizure -> %d total "
        "(ratio %.1f:1)",
        n_seizure, target_non_seizure, len(y_balanced), target_non_seizure / max(n_seizure, 1),
    )

    return X_balanced, y_balanced


# ── PCA and Scaling ───────────────────────────────────────────


def fit_pre_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit standard scaler before PCA on training data ONLY."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    logger.info("Pre-scaler (StandardScaler) fitted.")
    return scaler


def fit_pca(X_train: np.ndarray, n_components: int) -> PCA:
    """Fit PCA on training data ONLY."""
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    logger.info(
        "PCA fitted: %d -> %d components, explained variance=%.3f",
        X_train.shape[1], n_components,
        float(np.sum(pca.explained_variance_ratio_)),
    )
    return pca


def fit_scaler(
    X_train_pca: np.ndarray,
    feature_range: tuple[float, float],
) -> MinMaxScaler:
    """Fit scaler on training data ONLY."""
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(X_train_pca)
    logger.info(
        "Scaler (MinMaxScaler) fitted: range=(%.4f, %.4f)", feature_range[0], feature_range[1],
    )
    return scaler


def preprocess_features(
    X: np.ndarray,
    pre_scaler: StandardScaler,
    pca: PCA,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """Apply fitted pre-scaler, PCA, and final scaler to features."""
    X_scaled = pre_scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    X_final = scaler.transform(X_pca)
    return X_final


# ── Full Fold Preparation ────────────────────────────────────


def prepare_fold(
    fold: LOSOFold,
    feature_config: FeatureConfig,
    balancing_config: BalancingConfig,
    ratio: float = 1.0,
) -> PreparedFold:
    """Prepare one LOSO fold directly (mostly for smoke tests or standard pipeline).

    Note: In the full optimized training loop, we do inner validation and explicit fitting.
    """
    train_before = len(fold.y_train)
    train_seizure_before = int((fold.y_train == Label.SEIZURE).sum())
    train_non_before = int((fold.y_train == Label.NON_SEIZURE).sum())

    # Fit transformers on the FULL training fold
    pre_scaler = fit_pre_scaler(fold.X_train)
    X_train_prescaled = pre_scaler.transform(fold.X_train)
    
    pca = fit_pca(X_train_prescaled, feature_config.pca.n_components)
    X_train_pca = pca.transform(X_train_prescaled)
    
    scaler = fit_scaler(X_train_pca, feature_config.scaler.feature_range)

    # Balance the training data for VQC fit
    X_bal, y_bal = balance_training_data(fold.X_train, fold.y_train, balancing_config, ratio=ratio)

    # Apply transformers
    X_train_prepared = preprocess_features(X_bal, pre_scaler, pca, scaler)
    X_test_prepared = preprocess_features(fold.X_test, pre_scaler, pca, scaler)

    test_seizure = int((fold.y_test == Label.SEIZURE).sum())
    test_non = int((fold.y_test == Label.NON_SEIZURE).sum())

    return PreparedFold(
        fold_index=fold.fold_index,
        train_subjects=fold.train_subjects,
        test_subject=fold.test_subject,
        X_train_balanced=X_bal,
        y_train_balanced=y_bal,
        train_segments_before_balance=train_before,
        train_seizure_before_balance=train_seizure_before,
        train_nonseizure_before_balance=train_non_before,
        train_segments_after_balance=len(y_bal),
        X_train_prepared=X_train_prepared,
        X_test_prepared=X_test_prepared,
        y_train=y_bal,
        y_test=fold.y_test,
        test_segments=len(fold.y_test),
        test_seizure=test_seizure,
        test_nonseizure=test_non,
        pre_scaler=pre_scaler,
        pca=pca,
        scaler=scaler,
    )
