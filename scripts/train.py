"""Training entry point -- Quantum EEG Signal Classifier.

Phase 11: Single clear training script with checkpoint & resume support.

Usage:
    python scripts/train.py
    python scripts/train.py --config config/training.yaml

Methodology:
    1. Build dataset from CHB-MIT EDF files.
    2. Create outer LOSO folds (one held-out test subject per fold).
    3. For each outer fold, perform rotating inner LOSO validation
       across ALL training subjects to evaluate candidate configurations.
    4. Aggregate inner-validation metrics and select global optimal config.
    5. Evaluate the selected configuration on outer held-out test subjects
       (with atomic fold-level checkpointing & resume support).
    6. Train the production model on ALL subjects using the selected config
       (with atomic production-level checkpointing & resume support).
    7. Save all artifacts and evaluation reports.

Checkpoint/Resume:
    - Inner Grid Search fits are saved to grid_search_checkpoint.json.
    - Outer LOSO folds are saved atomically to checkpoints/fold_X/.
    - Production model training is saved atomically to checkpoints/production/.
    - Rerunning the script automatically resumes from the latest completed state.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_feature_config, load_training_config
from core.logging_setup import setup_logging
from core.models import FoldMetrics
from quantum.classifier import predict_vqc
from training.build_dataset import build_dataset
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
from training.evaluate import aggregate_fold_metrics, compute_fold_metrics
from training.prepare_dataset import (
    balance_training_data,
    create_loso_folds,
    fit_pca,
    fit_pre_scaler,
    fit_scaler,
    preprocess_features,
)
from training.save_artifacts import save_model_artifacts
from training.train_vqc import create_and_train_vqc

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Quantum EEG VQC model."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to training.yaml config file.",
    )
    parser.add_argument(
        "--features-config", type=str, default=None,
        help="Path to features.yaml config file.",
    )
    args = parser.parse_args()

    setup_logging(level=logging.INFO)

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("QUANTUM EEG SIGNAL CLASSIFIER -- TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── Load Configuration ──
    feature_config = load_feature_config(args.features_config)
    training_config = load_training_config(args.config)

    model_version = training_config.output.model_version
    artifacts_dir = PROJECT_ROOT / training_config.output.artifacts_dir
    version_dir = get_version_dir(artifacts_dir, model_version)
    checkpoint_dir = get_checkpoint_dir(artifacts_dir, model_version)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Feature config: %d channels, %d features/ch -> %d total",
                feature_config.n_channels, feature_config.n_features_per_channel,
                feature_config.total_features)
    logger.info("Training config: subjects=%s, evaluation=%s",
                training_config.dataset.subjects, training_config.evaluation_method)

    # ── Step 1-7: Build Dataset ──
    logger.info("=" * 60)
    logger.info("STEP 1-7: Building dataset...")
    logger.info("=" * 60)
    dataset = build_dataset(training_config, feature_config, PROJECT_ROOT)
    logger.info("Dataset: X=%s, y=%s, subjects=%s",
                dataset.X.shape, dataset.y.shape, dataset.subjects)

    # ── Step 8: Subject-Aware LOSO Splitting ──
    logger.info("=" * 60)
    logger.info("STEP 8: Creating LOSO folds...")
    logger.info("=" * 60)
    folds = create_loso_folds(dataset.X, dataset.y, dataset.metadata, dataset.subjects)
    logger.info("Created %d LOSO folds.", len(folds))

    # ── Initial Checkpoint Status Reporting ──
    completed_fold_indices = get_completed_folds(checkpoint_dir, len(folds))
    prod_status = get_production_status(checkpoint_dir, version_dir)

    logger.info("=" * 60)
    logger.info("CHECKPOINT STATUS")
    logger.info("  Training version: %s", model_version)
    logger.info("  Total LOSO folds: %d", len(folds))
    logger.info("  Completed checkpoints: %d/%d", len(completed_fold_indices), len(folds))
    logger.info("  Remaining folds: %d/%d", len(folds) - len(completed_fold_indices), len(folds))
    logger.info("  Production model: %s", prod_status)
    logger.info("=" * 60)

    try:
        # ── Hyperparameter Grid ──
        pca_dims_to_test = [4]
        ratios_to_test = [1.0, 2.0]
        thresholds_to_test = np.round(np.arange(0.1, 0.95, 0.1), 2)

        logger.info("=" * 60)
        logger.info("HYPERPARAMETER GRID SEARCH (Rotating Inner LOSO Validation)")
        logger.info("  PCA dims: %s", pca_dims_to_test)
        logger.info("  Class ratios: %s", ratios_to_test)
        logger.info("  Thresholds: %s", thresholds_to_test.tolist())
        expected_inner_fits = (
            len(folds)
            * sum(len(f.train_subjects) for f in folds) // len(folds)
            * len(pca_dims_to_test)
            * len(ratios_to_test)
        )
        logger.info("  Expected inner VQC fits: %d", expected_inner_fits)
        logger.info("=" * 60)

        # ── Checkpoint / Resume Support for Grid Search ──
        checkpoint_path = checkpoint_dir / "grid_search_checkpoint.json"
        checkpoint_data: dict = {"completed": [], "validation_results": {}}
        if checkpoint_path.exists():
            try:
                checkpoint_data = json.loads(
                    checkpoint_path.read_text(encoding="utf-8")
                )
                logger.info(
                    "Loaded grid search checkpoint: %d completed VQC fits",
                    len(checkpoint_data.get("completed", [])),
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Grid search checkpoint corrupt, starting fresh: %s", e)
                checkpoint_data = {"completed": [], "validation_results": {}}

        config_validation_results: dict[tuple, list[dict]] = {}
        completed_fits: set[tuple] = set()
        for entry in checkpoint_data.get("completed", []):
            completed_fits.add(tuple(entry["key"]))

        for key_str, metrics_list in checkpoint_data.get("validation_results", {}).items():
            config_key = tuple(json.loads(key_str))
            config_validation_results[config_key] = metrics_list

        grid_search_start = time.time()
        total_inner_vqc_fits = 0

        # ──────────────────────────────────────────────────────────────
        # PHASE 1: Inner Validation Grid Search
        # ──────────────────────────────────────────────────────────────
        for fold in folds:
            logger.info("-" * 40)
            logger.info("OUTER FOLD %d: test_subject=%s", fold.fold_index, fold.test_subject)

            for inner_val_subj in fold.train_subjects:
                inner_train_subjects = [s for s in fold.train_subjects if s != inner_val_subj]
                inner_val_mask = np.array([m.subject_id == inner_val_subj for m in fold.meta_train])
                inner_train_mask = ~inner_val_mask

                X_inner_train = fold.X_train[inner_train_mask]
                y_inner_train = fold.y_train[inner_train_mask]
                X_inner_val = fold.X_train[inner_val_mask]
                y_inner_val = fold.y_train[inner_val_mask]

                for pca_dim in pca_dims_to_test:
                    inner_pre_scaler = fit_pre_scaler(X_inner_train)
                    X_it_prescaled = inner_pre_scaler.transform(X_inner_train)

                    inner_pca = fit_pca(X_it_prescaled, pca_dim)
                    X_it_pca = inner_pca.transform(X_it_prescaled)

                    inner_scaler = fit_scaler(X_it_pca, feature_config.scaler.feature_range)
                    X_iv_prep = preprocess_features(
                        X_inner_val, inner_pre_scaler, inner_pca, inner_scaler
                    )

                    for ratio in ratios_to_test:
                        X_it_bal, y_it_bal = balance_training_data(
                            X_inner_train, y_inner_train,
                            training_config.balancing, ratio=ratio
                        )
                        X_it_bal_prep = preprocess_features(
                            X_it_bal, inner_pre_scaler, inner_pca, inner_scaler
                        )

                        fit_key = (fold.fold_index, inner_val_subj, pca_dim, ratio)
                        if fit_key in completed_fits:
                            total_inner_vqc_fits += 1
                            continue

                        fit_start = time.time()
                        temp_vqc = create_and_train_vqc(
                            training_config.vqc, X_it_bal_prep, y_it_bal,
                            num_qubits=pca_dim,
                        )
                        fit_elapsed = time.time() - fit_start
                        total_inner_vqc_fits += 1

                        proba = temp_vqc.predict_proba(X_iv_prep)
                        prob_positive = proba[:, 1] if (proba.ndim == 2 and proba.shape[1] >= 2) else proba.ravel()

                        for th in thresholds_to_test:
                            th_rounded = round(float(th), 2)
                            y_pred_th = (prob_positive >= th).astype(int)
                            metrics = compute_fold_metrics(y_inner_val, y_pred_th)

                            config_key = (pca_dim, ratio, th_rounded)
                            if config_key not in config_validation_results:
                                config_validation_results[config_key] = []

                            config_validation_results[config_key].append({
                                "f1": metrics.f1_score,
                                "precision": metrics.precision,
                                "recall": metrics.recall,
                                "sensitivity": metrics.sensitivity,
                                "outer_fold": fold.fold_index,
                                "inner_val_subj": inner_val_subj,
                            })

                        completed_fits.add(fit_key)
                        checkpoint_data["completed"].append({
                            "key": list(fit_key),
                            "pca_dim": pca_dim,
                            "ratio": ratio,
                            "outer_fold": fold.fold_index,
                            "inner_val_subj": inner_val_subj,
                            "samples": len(y_it_bal),
                            "elapsed_seconds": fit_elapsed,
                            "timestamp": datetime.now().isoformat(),
                        })
                        checkpoint_data["validation_results"] = {
                            json.dumps(list(k)): v
                            for k, v in config_validation_results.items()
                        }
                        checkpoint_path.write_text(
                            json.dumps(checkpoint_data, indent=2), encoding="utf-8"
                        )

        grid_search_elapsed = time.time() - grid_search_start
        logger.info("GRID SEARCH COMPLETE: %d VQC fits in %.1f seconds",
                    total_inner_vqc_fits, grid_search_elapsed)

        # ──────────────────────────────────────────────────────────────
        # PHASE 2: Select Global Optimal Configuration
        # ──────────────────────────────────────────────────────────────
        best_config = None
        best_score = (-1.0, -1.0, -1.0)

        for config_key, metrics_list in config_validation_results.items():
            avg_f1 = float(np.mean([m["f1"] for m in metrics_list]))
            avg_prec = float(np.mean([m["precision"] for m in metrics_list]))
            avg_sens = float(np.mean([m["sensitivity"] for m in metrics_list]))
            current_score = (avg_f1, avg_sens, avg_prec)
            if current_score > best_score:
                best_score = current_score
                best_config = config_key

        if best_config is None:
            raise RuntimeError("No valid configurations found during grid search.")

        best_pca_dim, best_ratio, best_threshold = best_config
        logger.info("=" * 60)
        logger.info("SELECTED GLOBAL OPTIMAL CONFIGURATION:")
        logger.info("  PCA Components (Qubits): %d", best_pca_dim)
        logger.info("  Class Ratio: %.1f:1", best_ratio)
        logger.info("  Decision Threshold: %.2f", best_threshold)
        logger.info("  Aggregated Inner-Val F1: %.4f", best_score[0])
        logger.info("=" * 60)

        # ──────────────────────────────────────────────────────────────
        # PHASE 3: Evaluate on Outer Held-Out Test Subjects (Fold Checkpoints)
        # ──────────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("EVALUATING SELECTED CONFIG ON OUTER HELD-OUT TEST SUBJECTS")
        logger.info("=" * 60)

        fold_metrics_list = []
        fold_results = []

        for fold in folds:
            fold_idx = fold.fold_index
            ckpt = load_fold_checkpoint(checkpoint_dir, fold_idx)

            if ckpt is not None:
                logger.info("Resuming %s training", model_version)
                logger.info("Skipping completed fold: Fold %d (test_subject: %s)", fold_idx, fold.test_subject)
                meta = ckpt["metadata"]
                m_dict = meta["metrics"]

                metrics_obj = FoldMetrics(
                    accuracy=m_dict["accuracy"],
                    precision=m_dict["precision"],
                    recall=m_dict["recall"],
                    f1_score=m_dict["f1_score"],
                    sensitivity=m_dict["sensitivity"],
                    specificity=m_dict["specificity"],
                    roc_auc=m_dict.get("roc_auc"),
                    confusion_matrix=m_dict["confusion_matrix"],
                )
                fold_metrics_list.append(metrics_obj)
                fold_results.append({
                    "fold": fold_idx,
                    "test_subject": fold.test_subject,
                    "train_subjects": fold.train_subjects,
                    "test_segments": len(fold.y_test),
                    "selected_pca_dim": best_pca_dim,
                    "selected_ratio": best_ratio,
                    "selected_threshold": float(best_threshold),
                    "metrics": m_dict,
                })
                logger.info(
                    "Loaded Fold %d checkpoint results: acc=%.3f prec=%.3f rec=%.3f f1=%.3f sens=%.3f spec=%.3f",
                    fold_idx, metrics_obj.accuracy, metrics_obj.precision,
                    metrics_obj.recall, metrics_obj.f1_score,
                    metrics_obj.sensitivity, metrics_obj.specificity,
                )
                continue

            logger.info("Starting fold: Fold %d (test_subject: %s)", fold_idx, fold.test_subject)

            fold_pre_scaler = fit_pre_scaler(fold.X_train)
            X_train_prescaled = fold_pre_scaler.transform(fold.X_train)

            fold_pca = fit_pca(X_train_prescaled, best_pca_dim)
            X_train_pca = fold_pca.transform(X_train_prescaled)

            fold_scaler = fit_scaler(X_train_pca, feature_config.scaler.feature_range)

            X_train_bal, y_train_bal = balance_training_data(
                fold.X_train, fold.y_train,
                training_config.balancing, ratio=best_ratio,
            )
            X_train_bal_prep = preprocess_features(
                X_train_bal, fold_pre_scaler, fold_pca, fold_scaler,
            )

            fold_vqc = create_and_train_vqc(
                training_config.vqc, X_train_bal_prep, y_train_bal,
                num_qubits=best_pca_dim,
            )

            X_test_prep = preprocess_features(
                fold.X_test, fold_pre_scaler, fold_pca, fold_scaler,
            )
            y_pred = predict_vqc(fold_vqc, X_test_prep, threshold=best_threshold)
            metrics = compute_fold_metrics(fold.y_test, y_pred)
            fold_metrics_list.append(metrics)

            m_dict = {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "sensitivity": metrics.sensitivity,
                "specificity": metrics.specificity,
                "confusion_matrix": metrics.confusion_matrix,
            }

            fold_results.append({
                "fold": fold_idx,
                "test_subject": fold.test_subject,
                "train_subjects": fold.train_subjects,
                "test_segments": len(fold.y_test),
                "selected_pca_dim": best_pca_dim,
                "selected_ratio": best_ratio,
                "selected_threshold": float(best_threshold),
                "metrics": m_dict,
            })

            fold_ckpt_dir = save_fold_checkpoint(
                checkpoint_dir=checkpoint_dir,
                fold_index=fold_idx,
                test_subject=fold.test_subject,
                train_subjects=fold.train_subjects,
                vqc=fold_vqc,
                pre_scaler=fold_pre_scaler,
                pca=fold_pca,
                scaler=fold_scaler,
                metrics=m_dict,
                selected_pca_dim=best_pca_dim,
                selected_ratio=best_ratio,
                selected_threshold=float(best_threshold),
                model_version=model_version,
                feature_schema_hash=feature_config.schema_hash(),
            )

            current_completed = get_completed_folds(checkpoint_dir, len(folds))
            logger.info("Fold %d completed", fold_idx)
            logger.info("Checkpoint saved: %s", fold_ckpt_dir)
            logger.info("Progress: %d/%d folds completed", len(current_completed), len(folds))

        # ── Aggregate Outer Test Metrics ──
        logger.info("=" * 60)
        logger.info("AGGREGATE OUTER TEST RESULTS")
        logger.info("=" * 60)
        agg = aggregate_fold_metrics(fold_metrics_list)

        # ──────────────────────────────────────────────────────────────
        # PHASE 4: Train Production Model on ALL Subjects
        # ──────────────────────────────────────────────────────────────
        prod_ckpt = load_production_checkpoint(checkpoint_dir, version_dir)

        if prod_ckpt is not None:
            logger.info("=" * 60)
            logger.info("PRODUCTION MODEL CHECKPOINT DETECTED (Status: COMPLETED)")
            logger.info("Production model already trained and validated for %s.", model_version)
            logger.info("Skipping production model retraining.")
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("TRAINING PRODUCTION MODEL on ALL subjects "
                        "(PCA=%d, ratio=%.1f, th=%.2f)...",
                        best_pca_dim, best_ratio, best_threshold)
            logger.info("=" * 60)

            prod_pre_scaler = fit_pre_scaler(dataset.X)
            X_prescaled = prod_pre_scaler.transform(dataset.X)

            prod_pca = fit_pca(X_prescaled, best_pca_dim)
            X_pca = prod_pca.transform(X_prescaled)

            prod_scaler = fit_scaler(X_pca, feature_config.scaler.feature_range)

            X_all_bal, y_all_bal = balance_training_data(
                dataset.X, dataset.y, training_config.balancing, ratio=best_ratio,
            )
            X_all_bal_prep = preprocess_features(
                X_all_bal, prod_pre_scaler, prod_pca, prod_scaler,
            )

            prod_vqc = create_and_train_vqc(
                training_config.vqc, X_all_bal_prep, y_all_bal,
                num_qubits=best_pca_dim,
            )

            saved_dir = save_model_artifacts(
                artifacts_dir=artifacts_dir,
                vqc=prod_vqc,
                pre_scaler=prod_pre_scaler,
                pca=prod_pca,
                scaler=prod_scaler,
                feature_config=feature_config,
                vqc_config=training_config.vqc,
                model_version=model_version,
                training_subjects=dataset.subjects,
                threshold=float(best_threshold),
                best_ratio=float(best_ratio),
                evaluation_metrics=agg,
            )

            eval_report = {
                "report_version": "2.0",
                "methodology": "nested_loso_with_rotating_inner_validation",
                "generated_at": datetime.now().isoformat(),
                "selected_configuration": {
                    "pca_dim": best_pca_dim,
                    "class_ratio": best_ratio,
                    "threshold": float(best_threshold),
                    "aggregated_inner_val_f1": best_score[0],
                    "aggregated_inner_val_sensitivity": best_score[1],
                    "aggregated_inner_val_precision": best_score[2],
                },
                "grid_search": {
                    "pca_dims_tested": pca_dims_to_test,
                    "ratios_tested": ratios_to_test,
                    "thresholds_tested": thresholds_to_test.tolist(),
                    "total_vqc_fits": total_inner_vqc_fits,
                    "grid_search_seconds": grid_search_elapsed,
                },
                "outer_test_folds": fold_results,
                "aggregate_outer_test_metrics": agg,
                "production_model_subjects": dataset.subjects,
                "production_balanced_samples": len(y_all_bal),
                "total_training_time_seconds": time.time() - start_time,
                "limitations": (
                    "Nested validation with only 3 subjects has high variance. "
                    "Inner validation metrics may not generalize."
                ),
            }
            report_path = version_dir / "evaluation_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
            logger.info("Saved evaluation report: %s", report_path)

            prod_meta = {
                "model_version": model_version,
                "training_subjects": dataset.subjects,
                "selected_pca_dim": best_pca_dim,
                "selected_ratio": best_ratio,
                "selected_threshold": float(best_threshold),
                "created_at": datetime.now().isoformat(),
            }
            save_production_checkpoint(
                checkpoint_dir=checkpoint_dir,
                vqc=prod_vqc,
                pre_scaler=prod_pre_scaler,
                pca=prod_pca,
                scaler=prod_scaler,
                metadata=prod_meta,
                eval_report=eval_report,
                version_dir=version_dir,
            )

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Training completed successfully")
        logger.info("All LOSO folds completed")
        logger.info("Final model saved: %s", version_dir)
        logger.info("Training time: %.1f seconds", elapsed)
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("=" * 60)
        logger.warning("TRAINING INTERRUPTED BY USER (Ctrl+C)")
        current_completed = get_completed_folds(checkpoint_dir, len(folds))
        curr_prod_status = get_production_status(checkpoint_dir, version_dir)
        logger.warning("Completed LOSO folds: %d/%d", len(current_completed), len(folds))
        logger.warning("Production model status: %s", curr_prod_status)
        logger.warning("Progress saved. Rerunning will resume from the latest checkpoint.")
        logger.warning("=" * 60)
        sys.exit(130)


if __name__ == "__main__":
    main()
