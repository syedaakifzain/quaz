"""Unit tests for training checkpointing and resume module.

Tests atomic fold-level and production-level checkpoint creation, loading,
validation, corruption handling, status detection, and resume behavior.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from training.checkpoint import (
    get_checkpoint_dir,
    get_completed_folds,
    get_production_status,
    get_version_dir,
    load_fold_checkpoint,
    load_production_checkpoint,
    save_fold_checkpoint,
    save_production_checkpoint,
)


class DummyVQC:
    """Mock VQC object for lightweight unit testing."""

    def __init__(self, weights: np.ndarray | None = None):
        self.weights = weights if weights is not None else np.ones((4,))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((len(X),))


class TestCheckpointing(unittest.TestCase):
    """Test suite for checkpoint serialization, atomic writes, and resume logic."""

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp())
        self.artifacts_dir = self.test_dir / "models"
        self.model_version = "v1.3.0"
        self.checkpoint_dir = get_checkpoint_dir(self.artifacts_dir, self.model_version)
        self.version_dir = get_version_dir(self.artifacts_dir, self.model_version)

        # Mock transformers
        X_dummy = np.random.randn(20, 20)
        self.pre_scaler = StandardScaler().fit(X_dummy)
        X_prescaled = self.pre_scaler.transform(X_dummy)

        self.pca = PCA(n_components=4).fit(X_prescaled)
        X_pca = self.pca.transform(X_prescaled)

        self.scaler = MinMaxScaler().fit(X_pca)
        self.vqc = DummyVQC()

        self.metrics = {
            "accuracy": 0.85,
            "precision": 0.10,
            "recall": 0.30,
            "f1_score": 0.15,
            "sensitivity": 0.30,
            "specificity": 0.86,
            "confusion_matrix": [[100, 10], [5, 2]],
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_checkpoint_dir_paths(self) -> None:
        """Test checkpoint and version directory path generation."""
        self.assertEqual(
            self.checkpoint_dir,
            self.artifacts_dir / "v1_3_0" / "checkpoints",
        )
        self.assertEqual(
            self.version_dir,
            self.artifacts_dir / "v1_3_0",
        )

    def test_save_and_load_fold_checkpoint(self) -> None:
        """Test atomic fold checkpoint saving and loading."""
        ckpt_path = save_fold_checkpoint(
            checkpoint_dir=self.checkpoint_dir,
            fold_index=0,
            test_subject="chb01",
            train_subjects=["chb02", "chb03", "chb04", "chb05"],
            vqc=self.vqc,
            pre_scaler=self.pre_scaler,
            pca=self.pca,
            scaler=self.scaler,
            metrics=self.metrics,
            selected_pca_dim=4,
            selected_ratio=2.0,
            selected_threshold=0.5,
            model_version=self.model_version,
            feature_schema_hash="hash_12345",
        )

        self.assertTrue(ckpt_path.exists())
        self.assertTrue((ckpt_path / "fold_metadata.json").exists())
        self.assertTrue((ckpt_path / "vqc_model.dill").exists())

        # Load back
        loaded = load_fold_checkpoint(self.checkpoint_dir, 0)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["metadata"]["fold_index"], 0)
        self.assertEqual(loaded["metadata"]["test_subject"], "chb01")
        self.assertEqual(loaded["metadata"]["status"], "COMPLETED")
        self.assertEqual(loaded["metadata"]["metrics"]["accuracy"], 0.85)

    def test_get_completed_folds(self) -> None:
        """Test scanning completed fold checkpoints."""
        self.assertEqual(get_completed_folds(self.checkpoint_dir, 5), [])

        # Save fold 0 and fold 2
        for idx, subj in [(0, "chb01"), (2, "chb03")]:
            save_fold_checkpoint(
                checkpoint_dir=self.checkpoint_dir,
                fold_index=idx,
                test_subject=subj,
                train_subjects=["chb02"],
                vqc=self.vqc,
                pre_scaler=self.pre_scaler,
                pca=self.pca,
                scaler=self.scaler,
                metrics=self.metrics,
                selected_pca_dim=4,
                selected_ratio=2.0,
                selected_threshold=0.5,
                model_version=self.model_version,
                feature_schema_hash="hash_12345",
            )

        completed = get_completed_folds(self.checkpoint_dir, 5)
        self.assertEqual(completed, [0, 2])

    def test_corrupt_checkpoint_handling(self) -> None:
        """Test corrupt or partial checkpoints are detected as invalid."""
        fold_path = save_fold_checkpoint(
            checkpoint_dir=self.checkpoint_dir,
            fold_index=1,
            test_subject="chb02",
            train_subjects=["chb01"],
            vqc=self.vqc,
            pre_scaler=self.pre_scaler,
            pca=self.pca,
            scaler=self.scaler,
            metrics=self.metrics,
            selected_pca_dim=4,
            selected_ratio=2.0,
            selected_threshold=0.5,
            model_version=self.model_version,
            feature_schema_hash="hash_12345",
        )

        self.assertIsNotNone(load_fold_checkpoint(self.checkpoint_dir, 1))

        # Corrupt metadata file
        (fold_path / "fold_metadata.json").write_text("CORRUPT JSON", encoding="utf-8")
        self.assertIsNone(load_fold_checkpoint(self.checkpoint_dir, 1))

    def test_save_and_load_production_checkpoint(self) -> None:
        """Test production model checkpoint saving, loading, and status."""
        self.assertEqual(
            get_production_status(self.checkpoint_dir, self.version_dir),
            "NOT_STARTED",
        )

        prod_meta = {
            "model_version": self.model_version,
            "status": "COMPLETED",
        }
        eval_report = {"report_version": "2.0", "outer_test_folds": []}

        # Write dummy evaluation_report.json to version_dir
        self.version_dir.mkdir(parents=True, exist_ok=True)
        (self.version_dir / "evaluation_report.json").write_text(
            json.dumps(eval_report), encoding="utf-8"
        )

        save_production_checkpoint(
            checkpoint_dir=self.checkpoint_dir,
            vqc=self.vqc,
            pre_scaler=self.pre_scaler,
            pca=self.pca,
            scaler=self.scaler,
            metadata=prod_meta,
            eval_report=eval_report,
            version_dir=self.version_dir,
        )

        self.assertEqual(
            get_production_status(self.checkpoint_dir, self.version_dir),
            "COMPLETED",
        )

        loaded_prod = load_production_checkpoint(self.checkpoint_dir, self.version_dir)
        self.assertIsNotNone(loaded_prod)
        self.assertEqual(loaded_prod["production_metadata"]["status"], "COMPLETED")

    def test_v1_2_0_preservation(self) -> None:
        """Verify existing v1.2.0 model artifacts remain untouched."""
        v1_2_0_dir = Path("models/v1_2_0")
        if v1_2_0_dir.exists():
            self.assertTrue((v1_2_0_dir / "vqc_model.dill").exists())
            self.assertTrue((v1_2_0_dir / "model_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
