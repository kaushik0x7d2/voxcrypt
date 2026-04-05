"""
FHE Inference Client — encrypts audio features and sends to the server.

Production-hardened client with API key auth, retry logic, circuit breaker,
and structured logging.

Usage:
    python demo/client.py
    python demo/client.py --url http://server:5000 --api-key your-key
"""

import os
import sys
import time
import base64
import argparse

import torch
import numpy as np
import requests as http_requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "orion", "repo"))

import orion
from orion.backend.python.tensors import CipherTensor
from speaker_verify.logging_config import setup_logging, get_logger
from speaker_verify.resilience import retry, CircuitBreaker
from speaker_verify.security import decrypt_key_file

logger = get_logger("client")


def setup_client(config_path, sk_path, key_password=None):
    """
    Initialize the client's FHE scheme and load the server's secret key.
    """
    scheme = orion.init_scheme(config_path)

    with open(sk_path, "rb") as f:
        sk_bytes = f.read()

    # Decrypt key if password provided
    if key_password:
        sk_bytes = decrypt_key_file(sk_bytes, key_password)

    sk_arr = np.frombuffer(sk_bytes, dtype=np.uint8)
    scheme.backend.LoadSecretKey(sk_arr)

    scheme.backend.GeneratePublicKey()
    scheme.backend.NewEncryptor()
    scheme.backend.NewDecryptor()

    return scheme


def encrypt_sample(sample_data, input_level):
    """Encrypt a feature vector and serialize for transport."""
    ptxt = orion.encode(sample_data, input_level)
    ctxt = orion.encrypt(ptxt)
    serialized = ctxt.serialize()
    return {
        "ciphertexts": [
            base64.b64encode(b).decode() for b in serialized["ciphertexts"]
        ],
        "shape": serialized["shape"],
        "on_shape": serialized["on_shape"],
    }


@retry(
    max_retries=3,
    backoff_base=2.0,
    retryable=(ConnectionError, http_requests.ConnectionError, http_requests.Timeout),
)
def send_for_inference(server_url, payload, api_key=None, timeout=120):
    """Send encrypted ciphertext to the server and get back encrypted result."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = http_requests.post(
        f"{server_url}/api/v1/predict", json=payload, headers=headers, timeout=timeout
    )

    if resp.status_code == 401:
        raise AuthError("Authentication failed. Check your API key.")
    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After", "60")
        raise RateLimited(f"Rate limited. Retry after {retry_after}s")
    if resp.status_code == 503:
        raise ConnectionError("Server not ready")

    resp.raise_for_status()
    return resp.json()


def decrypt_result(scheme, response):
    """Deserialize and decrypt the server's response."""
    ct_data = {
        "ciphertexts": [base64.b64decode(b) for b in response["ciphertexts"]],
        "shape": response["shape"],
        "on_shape": response["on_shape"],
    }
    result_ctxt = CipherTensor.from_serialized(scheme, ct_data)
    result_ptxt = result_ctxt.decrypt()
    return result_ptxt.decode()


class AuthError(Exception):
    pass


class RateLimited(Exception):
    pass


def check_server_health(server_url, timeout=10):
    """Check if the server is healthy before starting batch inference."""
    try:
        resp = http_requests.get(f"{server_url}/health", timeout=timeout)
        data = resp.json()
        return data.get("status") == "healthy"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="FHE Speaker Verification Client")
    parser.add_argument("--url", default="http://127.0.0.1:5000", help="Server URL")
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of pairs to test"
    )
    parser.add_argument("--api-key", default=None, help="API key for authentication")
    parser.add_argument(
        "--key-password", default=None, help="Password to decrypt the secret key file"
    )
    args = parser.parse_args()

    # Get API key from env if not passed as arg
    api_key = args.api_key or os.environ.get("ORION_VOICE_API_KEY")

    setup_logging(level="INFO", fmt="text")

    torch.manual_seed(42)
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(demo_dir, "..", "configs", "fhe_config.yml")
    sk_path = os.path.join(demo_dir, "keys", "secret.key")

    if not os.path.exists(sk_path):
        logger.error(f"Missing {sk_path}. Start the server first.")
        return

    # Health check
    logger.info(f"Checking server health at {args.url}...")
    if not check_server_health(args.url):
        logger.error("Server is not healthy or unreachable.")
        return
    logger.info("Server is healthy.")

    # Setup circuit breaker for repeated failures
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30, name="server")

    logger.info("Initializing FHE scheme and loading server key...")
    t0 = time.time()
    scheme = setup_client(config_path, sk_path, key_password=args.key_password)
    logger.info(f"Ready ({time.time() - t0:.2f}s)")

    # Get input_level from server
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    info_resp = http_requests.get(
        f"{args.url}/api/v1/info", headers=headers, timeout=10
    ).json()
    input_level = info_resp["input_level"]
    model_version = info_resp.get("model_version", "unknown")
    logger.info(f"Server input level: {input_level}, model version: {model_version}")

    # Load test samples
    samples = np.load(os.path.join(demo_dir, "test_samples.npz"))
    X_test, y_test = samples["X"], samples["y"]

    labels = ["Diff Speaker", "Same Speaker"]
    results = []

    print(f"\n{'=' * 60}")
    print(f"  Private Speaker Verification ({args.num_samples} pairs)")
    print(f"  Server: {args.url}")
    print(f"  Auth: {'enabled' if api_key else 'disabled'}")
    print(f"{'=' * 60}\n")

    for i in range(min(args.num_samples, len(X_test))):
        sample_data = torch.tensor(X_test[i : i + 1], dtype=torch.float32)
        actual = int(y_test[i])

        # Encrypt locally
        t0 = time.time()
        payload = encrypt_sample(sample_data, input_level)
        t_enc = time.time() - t0

        ct_size = sum(len(base64.b64decode(b)) for b in payload["ciphertexts"])

        # Send to server (with circuit breaker)
        try:
            t0 = time.time()
            response = cb.call(send_for_inference, args.url, payload, api_key)
            t_server = time.time() - t0
        except Exception as e:
            logger.error(f"Pair {i + 1}: Server error: {e}")
            continue

        t_inf = response.get("inference_time", 0)

        # Decrypt locally
        t0 = time.time()
        out_values = decrypt_result(scheme, response)
        t_dec = time.time() - t0

        pred = out_values[:2].argmax().item()
        results.append((pred, actual))

        status = "correct" if pred == actual else "WRONG"
        print(
            f"  Pair {i + 1}: {labels[pred]:>14s}  "
            f"(actual: {labels[actual]:>14s}) [{status}]"
        )
        print(
            f"    Encrypt: {t_enc:.3f}s | "
            f"Network: {t_server:.3f}s (inference: {t_inf:.3f}s) | "
            f"Decrypt: {t_dec:.3f}s | "
            f"Ciphertext: {ct_size / 1024:.0f} KB"
        )

    if results:
        correct = sum(1 for p, a in results if p == a)
        total = len(results)
        print(f"\n{'=' * 60}")
        print(f"  FHE Accuracy: {correct}/{total} ({correct / total:.0%})")
        print(f"  The server verified {total} speaker pairs without")
        print("  ever hearing the audio or seeing the result.")
        print(f"{'=' * 60}")

    scheme.delete_scheme()


if __name__ == "__main__":
    main()
