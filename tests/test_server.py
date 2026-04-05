"""Tests for server endpoints."""

import pytest
import sys
import os
import importlib.util
from unittest.mock import MagicMock

# Load demo.server from the correct path (not orion/repo/demo/server.py)
_demo_server_path = os.path.join(os.path.dirname(__file__), "..", "demo", "server.py")
_spec = importlib.util.spec_from_file_location("demo_server", _demo_server_path)
_demo_server = importlib.util.module_from_spec(_spec)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "orion", "repo"))

pytest.importorskip("orion")

from speaker_verify.config import Config  # noqa: E402
from speaker_verify.logging_config import setup_logging  # noqa: E402

# Now execute the module
_spec.loader.exec_module(_demo_server)
create_app = _demo_server.create_app
_state = _demo_server._state


@pytest.fixture
def app():
    """Create a test Flask app without FHE initialization."""
    setup_logging(level="WARNING", fmt="text")
    config = Config()
    config.security.api_key = "test-key"
    config.security.require_auth = True

    app = create_app(config)
    _state._ready = True
    _state.model = MagicMock()  # ready property requires model is not None
    _state.input_level = 13
    _state.model_version = "test-v1"

    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestHealthEndpoints:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] in ("healthy", "starting")

    def test_readiness(self, client):
        resp = client.get("/api/v1/health/ready")
        assert resp.status_code == 200

    def test_liveness(self, client):
        resp = client.get("/api/v1/health/live")
        assert resp.status_code == 200
        assert resp.get_json()["alive"] is True


class TestInfoEndpoint:
    def test_info_without_auth(self, client):
        resp = client.get("/info")
        assert resp.status_code == 401

    def test_info_with_auth(self, client):
        resp = client.get("/info", headers={"Authorization": "Bearer test-key"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "input_level" in data

    def test_info_wrong_key(self, client):
        resp = client.get("/info", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401


class TestPredictEndpoint:
    def test_predict_no_auth(self, client):
        resp = client.post(
            "/predict",
            json={"ciphertexts": ["dGVzdA=="], "shape": [1], "on_shape": [1]},
        )
        assert resp.status_code == 401

    def test_predict_invalid_payload(self, client):
        resp = client.post(
            "/predict",
            json={"shape": [1]},
            headers={"Authorization": "Bearer test-key"},
        )
        assert resp.status_code == 400

    def test_predict_empty_body(self, client):
        resp = client.post(
            "/predict",
            data="not json",
            content_type="text/plain",
            headers={"Authorization": "Bearer test-key"},
        )
        assert resp.status_code == 400


class TestMetrics:
    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert b"uptime_seconds" in resp.data


class TestErrorResponses:
    def test_404(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
        data = resp.get_json()
        assert "code" in data
