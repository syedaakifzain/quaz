"""FastAPI backend server for Quantum EEG Signal Classifier.

Exposes REST endpoints for health status, EDF file analysis, and demo analysis.
Uses existing InferenceService pipeline without altering model weights or thresholds.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference.pipeline import InferenceService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Workspace root
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_VERSION = "v1_3_0"

from contextlib import asynccontextmanager

# Singleton InferenceService instance
inference_service: InferenceService | None = None


def startup_event():
    global inference_service
    try:
        logger.info("Initializing InferenceService with model %s from %s...", MODEL_VERSION, MODELS_DIR)
        inference_service = InferenceService(artifacts_dir=MODELS_DIR, model_version=MODEL_VERSION)
        logger.info("InferenceService initialized successfully (threshold=%.2f)", inference_service.threshold)
    except Exception as e:
        logger.error("Failed to initialize InferenceService: %s", e, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_event()
    yield


app = FastAPI(
    title="Quantum EEG Signal Classifier API",
    description="Backend API providing real VQC inference on EEG recordings",
    version="1.3.0",
    lifespan=lifespan,
)

# Enable CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def get_health() -> dict[str, Any]:
    """Return backend status and loaded model metadata."""
    if inference_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialized",
        )
    
    meta = inference_service.metadata
    return {
        "status": "ok",
        "model_version": inference_service.model_version,
        "classifier": meta.get("classifier", "VQC"),
        "threshold": inference_service.threshold,
        "qubits": meta.get("qubits", 4),
        "features_before_pca": meta.get("feature_count_before_pca", 20),
        "pca_components": meta.get("selected_pca_dim", 4),
        "channels": inference_service.feature_config.channels,
        "segment_duration_seconds": inference_service.feature_config.segment_duration_seconds,
        "sampling_frequency": 256.0,
    }


@app.post("/api/analyze")
async def analyze_edf(file: UploadFile = File(...)) -> dict[str, Any]:
    """Run real VQC inference on uploaded EDF file."""
    if inference_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialized",
        )

    filename = file.filename or ""
    if not filename.lower().endswith(".edf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only European Data Format (.edf) files are supported.",
        )

    # Save to temp file
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            shutil.copyfileobj(file.file, temp_file)
        except Exception as e:
            logger.error("Failed to save uploaded file: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process uploaded file: {str(e)}",
            )

    try:
        logger.info("Running inference on uploaded file: %s (temp: %s)", filename, temp_path)
        analysis_result = inference_service.predict(temp_path)
        result_dict = asdict(analysis_result)
        # Ensure original filename is presented
        result_dict["recording_name"] = filename
        return result_dict
    except Exception as e:
        logger.error("Inference failure for %s: %s", filename, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}",
        )
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/api/analyze-demo")
def analyze_demo() -> dict[str, Any]:
    """Run real VQC inference on demo EDF file (test_data/chb06_01.edf)."""
    if inference_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service not initialized",
        )

    # Check available demo files
    demo_files = [
        BASE_DIR / "test_data" / "chb06_01.edf",
        BASE_DIR / "test_data" / "chb07_01.edf",
    ]
    
    demo_path = None
    for df in demo_files:
        if df.exists():
            demo_path = df
            break
            
    if demo_path is None:
        raise HTTPException(
            status_code=status.HTTP_444_NOT_FOUND if hasattr(status, "HTTP_444_NOT_FOUND") else 404,
            detail="Demo EDF recording not found in test_data directory",
        )

    try:
        logger.info("Executing demo inference on %s", demo_path)
        analysis_result = inference_service.predict(demo_path)
        return asdict(analysis_result)
    except Exception as e:
        logger.error("Demo inference failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Demo inference error: {str(e)}",
        )
