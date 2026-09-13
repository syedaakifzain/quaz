"""Unit tests for FastAPI backend REST endpoints (api/main.py)."""

from __future__ import annotations

import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from api.main import app, startup_event, inference_service


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for FastAPI health status and analysis endpoints."""

    @classmethod
    def setUpClass(cls) -> None:
        startup_event()
        cls.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        """Test GET /api/health returns status ok and model metadata."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["model_version"], "v1_3_0")
        self.assertEqual(data["classifier"], "VQC")
        self.assertEqual(data["threshold"], 0.6)
        self.assertEqual(data["qubits"], 4)
        self.assertEqual(data["features_before_pca"], 20)
        self.assertEqual(data["pca_components"], 4)
        self.assertIn("channels", data)

    def test_invalid_file_extension(self) -> None:
        """Test POST /api/analyze rejects non-EDF files."""
        files = {"file": ("test.txt", b"invalid data", "text/plain")}
        response = self.client.post("/api/analyze", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid file format", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
