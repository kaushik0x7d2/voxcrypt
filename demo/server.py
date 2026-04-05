"""
FHE Inference Server — runs encrypted speaker verification.

Production-hardened server with API key auth, rate limiting, input validation,
structured logging, health checks, metrics, inference queue, and graceful shutdown.

Usage:
    # Development
    python demo/server.py

    # Production
    ORION_VOICE_API_KEY=your-secret-key \
    ORION_VOICE_REQUIRE_AUTH=true \
    ORION_VOICE_LOG_LEVEL=INFO \
    python demo/server.py --production
"""

import os
import sys
import time
import base64
import gc
import signal
import argparse
import uuid
import threading

import torch
import numpy as np
from flask import Flask, request, jsonify
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "orion", "repo"))

import orion
from orion.backend.python.tensors import CipherTensor
from speaker_verify.model import SpeakerVerifyNet
from speaker_verify.config import Config
from speaker_verify.logging_config import (
    setup_logging, get_logger, set_request_id, get_request_id)
from speaker_verify.metrics import registry
from speaker_verify.security import APIKeyAuth, RateLimiter, InputValidator
from speaker_verify.error_handlers import (
    register_error_handlers, AuthenticationError, RateLimitError,
    ValidationError, FHEInferenceError, ServiceUnavailableError)
from speaker_verify.resilience import InferenceQueue
from speaker_verify.artifacts import ModelManifest


# --- Server State ---

class ServerState:
    """Thread-safe server state."""
    def __init__(self):
        self._lock = threading.RLock()
        self.model = None
        self.scheme = None
        self.input_level = None
        self.start_time = time.time()
        self.model_version = "unknown"
        self._ready = False

    @property
    def ready(self):
        return self._ready and self.model is not None

    def set_ready(self):
        with self._lock:
            self._ready = True


_state = ServerState()
_config = None
_auth = None
_rate_limiter = None
_validator = None
_inference_queue = None
logger = None


# --- Application Factory ---

def create_app(config):
    """Create and configure the Flask application."""
    global _config, _auth, _rate_limiter, _validator, _inference_queue, logger

    _config = config
    logger = get_logger("server")

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.server.max_payload_mb * 1024 * 1024

    # Security
    _auth = APIKeyAuth(config.security.api_key, config.security.require_auth)
    _rate_limiter = RateLimiter(
        config.security.rate_limit, config.security.rate_limit_window)
    _validator = InputValidator(config.server.max_payload_mb)

    # Inference queue
    _inference_queue = InferenceQueue(max_queue_size=100, timeout=config.server.request_timeout)
    _inference_queue.start()

    # Register error handlers
    register_error_handlers(app)

    # --- Middleware ---

    @app.before_request
    def before_request():
        # Assign request ID
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        set_request_id(rid)
        registry.requests_total.inc()
        registry.active_requests.inc()

    @app.after_request
    def after_request(response):
        registry.active_requests.dec()
        rid = get_request_id()
        if rid:
            response.headers["X-Request-ID"] = rid
        return response

    # --- Endpoints ---

    @app.route("/api/v1/health", methods=["GET"])
    @app.route("/health", methods=["GET"])
    def health():
        """Health check — no auth required."""
        status = "healthy" if _state.ready else "starting"
        return jsonify({
            "status": status,
            "uptime_seconds": round(time.time() - _state.start_time, 1),
            "fhe_scheme_ready": _state.scheme is not None,
            "model_loaded": _state.model is not None,
            "model_version": _state.model_version,
            "active_requests": registry.active_requests.value,
            "queue_pending": _inference_queue.pending if _inference_queue else 0,
        })

    @app.route("/api/v1/health/ready", methods=["GET"])
    def ready():
        """Readiness probe."""
        if _state.ready:
            return jsonify({"ready": True}), 200
        return jsonify({"ready": False}), 503

    @app.route("/api/v1/health/live", methods=["GET"])
    def live():
        """Liveness probe."""
        return jsonify({"alive": True}), 200

    @app.route("/api/v1/metrics", methods=["GET"])
    @app.route("/metrics", methods=["GET"])
    def metrics():
        """Metrics endpoint."""
        return registry.to_text(), 200, {"Content-Type": "text/plain"}

    @app.route("/api/v1/info", methods=["GET"])
    @app.route("/info", methods=["GET"])
    def info():
        """Return scheme parameters the client needs."""
        ok, err = _auth.authenticate(request)
        if not ok:
            raise AuthenticationError(err)

        if not _state.ready:
            raise ServiceUnavailableError("Server is still starting")

        return jsonify({
            "input_level": _state.input_level,
            "status": "ready",
            "model_version": _state.model_version,
        })

    @app.route("/api/v1/predict", methods=["POST"])
    @app.route("/predict", methods=["POST"])
    def predict():
        """Run FHE inference on an encrypted ciphertext."""
        # Auth
        ok, err = _auth.authenticate(request)
        if not ok:
            raise AuthenticationError(err)

        # Rate limit
        ok, remaining_or_retry = _rate_limiter.allow(request.remote_addr)
        if not ok:
            raise RateLimitError(retry_after=remaining_or_retry)

        # Readiness
        if not _state.ready:
            raise ServiceUnavailableError("Server is still starting")

        # Validate input
        data = request.get_json(silent=True)
        ok, err = _validator.validate_predict_request(data)
        if not ok:
            raise ValidationError(err)

        # Track ciphertext size
        ct_size = sum(len(base64.b64decode(b)) for b in data["ciphertexts"])
        registry.ciphertext_size.observe(ct_size)

        logger.info("Inference request received",
                     extra={"extra_data": {
                         "ciphertext_size_kb": round(ct_size / 1024, 1),
                         "remote_addr": request.remote_addr}})

        # Run inference through queue
        def do_inference():
            ct_data = {
                "ciphertexts": [base64.b64decode(b)
                                for b in data["ciphertexts"]],
                "shape": data["shape"],
                "on_shape": data["on_shape"],
            }
            ctxt = CipherTensor.from_serialized(_state.scheme, ct_data)

            t0 = time.time()
            out_ctxt = _state.model(ctxt)
            t_inf = time.time() - t0

            result = out_ctxt.serialize()

            del ctxt, out_ctxt
            gc.collect()

            return result, t_inf

        try:
            future = _inference_queue.submit(do_inference)
            result, t_inf = future.result(
                timeout=_config.server.request_timeout)
        except TimeoutError:
            registry.inference_errors.inc()
            raise FHEInferenceError("Inference timed out")
        except Exception as e:
            registry.inference_errors.inc()
            logger.exception("Inference failed")
            raise FHEInferenceError(f"Inference failed: {type(e).__name__}")

        registry.inference_total.inc()
        registry.inference_duration.observe(t_inf)

        response = {
            "ciphertexts": [base64.b64encode(b).decode()
                            for b in result["ciphertexts"]],
            "shape": result["shape"],
            "on_shape": result["on_shape"],
            "inference_time": round(t_inf, 3),
            "model_version": _state.model_version,
            "request_id": get_request_id(),
        }

        logger.info(f"Inference completed in {t_inf:.3f}s")

        return jsonify(response)

    return app


# --- Startup ---

def startup(demo_dir, config):
    """Initialize the FHE scheme, load/compile model, export secret key."""
    config_path = config.fhe.config_path or os.path.join(
        demo_dir, "..", "configs", "fhe_config.yml")

    for f in ["speaker_model.pt", "scaler.npz", "test_samples.npz"]:
        path = os.path.join(demo_dir, f)
        if not os.path.exists(path):
            logger.error(f"Missing {f}. Run train_model.py first.")
            sys.exit(1)

    # Load model
    model = SpeakerVerifyNet()
    model.load_state_dict(torch.load(
        os.path.join(demo_dir, "speaker_model.pt"), weights_only=True))
    model.eval()

    # Load model version
    manifest_path = os.path.join(demo_dir, "model_manifest.json")
    if os.path.exists(manifest_path):
        manifest = ModelManifest.load(manifest_path)
        _state.model_version = manifest.version

        # Verify integrity
        ok, errors = manifest.verify_integrity(
            os.path.join(demo_dir, "speaker_model.pt"),
            os.path.join(demo_dir, "scaler.npz"))
        if not ok:
            for err in errors:
                logger.warning(f"Integrity check: {err}")
    else:
        logger.info("No model manifest found, skipping integrity check")

    # Init FHE scheme
    logger.info("Initializing FHE scheme...")
    t0 = time.time()
    scheme = orion.init_scheme(config_path)
    logger.info(f"Scheme ready ({time.time()-t0:.2f}s)")

    # Export secret key
    keys_dir = os.path.join(demo_dir, "keys")
    os.makedirs(keys_dir, exist_ok=True)
    import ctypes
    sk_arr, sk_ptr = scheme.backend.SerializeSecretKey()
    sk_bytes = bytes(sk_arr)
    scheme.backend.FreeCArray(ctypes.cast(sk_ptr, ctypes.c_void_p))

    # Optionally encrypt the key at rest
    sk_path = os.path.join(keys_dir, "secret.key")
    if config.security.key_password:
        from speaker_verify.security import encrypt_key_file
        encrypted = encrypt_key_file(sk_bytes, config.security.key_password)
        with open(sk_path, "wb") as f:
            f.write(encrypted)
        logger.info(f"Secret key exported (encrypted) to {sk_path}")
    else:
        with open(sk_path, "wb") as f:
            f.write(sk_bytes)
        logger.info(f"Secret key exported to {sk_path} ({len(sk_bytes)} bytes)")

    # Fit model
    logger.info("Fitting model for FHE...")
    t0 = time.time()
    samples = np.load(os.path.join(demo_dir, "test_samples.npz"))
    fit_X = torch.tensor(samples["X"], dtype=torch.float32)
    fit_dataset = TensorDataset(fit_X, torch.zeros(len(fit_X)))
    orion.fit(model, DataLoader(fit_dataset, batch_size=32))
    logger.info(f"Fit done ({time.time()-t0:.2f}s)")

    # Compile
    logger.info("Compiling model for FHE...")
    t0 = time.time()
    input_level = orion.compile(model)
    logger.info(f"Compiled ({time.time()-t0:.2f}s) | Input level: {input_level}")

    # Switch to HE mode
    model.he()

    _state.model = model
    _state.scheme = scheme
    _state.input_level = input_level
    _state.set_ready()

    logger.info("Server ready to accept encrypted inference requests.")


# --- Graceful Shutdown ---

def graceful_shutdown(signum=None, frame=None):
    """Clean up resources on shutdown."""
    if logger:
        logger.info("Shutting down...")

    if _inference_queue:
        _inference_queue.stop()

    if _state.scheme:
        _state.scheme.delete_scheme()
        if logger:
            logger.info("FHE scheme cleaned up")

    if signum:
        sys.exit(0)


# --- Main ---

def main():
    global logger

    parser = argparse.ArgumentParser(
        description="FHE Speaker Verification Server")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--production", action="store_true",
                        help="Use production WSGI server")
    args = parser.parse_args()

    torch.manual_seed(42)

    # Load configuration
    config = Config.from_env()
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.production:
        config.server.production = True

    config.validate()

    # Setup logging
    setup_logging(
        level=config.logging.level,
        fmt=config.logging.format,
        log_file=config.logging.log_file)
    logger = get_logger("server")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

    demo_dir = os.path.dirname(os.path.abspath(__file__))

    # Startup (FHE init, model load)
    startup(demo_dir, config)

    # Create app
    app = create_app(config)

    # Run
    if config.server.production:
        try:
            from waitress import serve
            logger.info(
                f"Starting production server on "
                f"{config.server.host}:{config.server.port}")
            serve(app, host=config.server.host, port=config.server.port,
                  threads=config.server.workers,
                  channel_timeout=config.server.request_timeout)
        except ImportError:
            logger.warning(
                "waitress not installed, falling back to Flask dev server. "
                "Install with: pip install waitress")
            app.run(host=config.server.host, port=config.server.port,
                    debug=False, threaded=True)
    else:
        ssl_ctx = None
        if config.security.tls_cert and config.security.tls_key:
            ssl_ctx = (config.security.tls_cert, config.security.tls_key)

        logger.info(
            f"Starting dev server on "
            f"{config.server.host}:{config.server.port}")
        app.run(host=config.server.host, port=config.server.port,
                debug=False, threaded=True, ssl_context=ssl_ctx)


if __name__ == "__main__":
    main()
