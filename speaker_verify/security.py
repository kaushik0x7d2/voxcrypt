"""
Security middleware for Orion Voice.

Provides API key authentication, rate limiting, input validation,
and key management utilities.
"""

import base64
import hashlib
import hmac
import time
import threading
from functools import wraps

from speaker_verify.logging_config import get_logger
from speaker_verify.metrics import registry

logger = get_logger("security")


# --- API Key Authentication ---

class APIKeyAuth:
    """API key authentication via Bearer token."""

    def __init__(self, api_key, require=False):
        self.api_key = api_key
        self.require = require

    def authenticate(self, request):
        """
        Verify the request's API key.

        Returns:
            (True, None) on success or if auth not required.
            (False, error_message) on failure.
        """
        if not self.require:
            return True, None

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            registry.auth_failures.inc()
            logger.warning("Missing Bearer token",
                           extra={"extra_data": {
                               "remote_addr": request.remote_addr}})
            return False, "Missing Authorization header"

        token = auth_header[7:]
        if not hmac.compare_digest(token, self.api_key):
            registry.auth_failures.inc()
            logger.warning("Invalid API key",
                           extra={"extra_data": {
                               "remote_addr": request.remote_addr}})
            return False, "Invalid API key"

        return True, None


# --- Rate Limiter ---

class RateLimiter:
    """Per-IP token bucket rate limiter."""

    def __init__(self, max_requests=10, window_seconds=60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets = {}
        self._lock = threading.Lock()

    def _cleanup(self):
        """Remove expired entries."""
        now = time.time()
        expired = [ip for ip, (_, ts) in self._buckets.items()
                   if now - ts > self.window * 2]
        for ip in expired:
            del self._buckets[ip]

    def allow(self, ip):
        """
        Check if a request from this IP is allowed.

        Returns:
            (True, remaining) if allowed.
            (False, retry_after) if rate limited.
        """
        now = time.time()

        with self._lock:
            if len(self._buckets) > 10000:
                self._cleanup()

            if ip not in self._buckets:
                self._buckets[ip] = (1, now)
                return True, self.max_requests - 1

            count, window_start = self._buckets[ip]

            if now - window_start > self.window:
                self._buckets[ip] = (1, now)
                return True, self.max_requests - 1

            if count >= self.max_requests:
                retry_after = int(self.window - (now - window_start)) + 1
                registry.rate_limit_hits.inc()
                logger.warning("Rate limit exceeded",
                               extra={"extra_data": {
                                   "ip": ip, "count": count}})
                return False, retry_after

            self._buckets[ip] = (count + 1, window_start)
            return True, self.max_requests - count - 1


# --- Input Validation ---

class InputValidator:
    """Validates request payloads."""

    def __init__(self, max_payload_mb=50):
        self.max_payload_bytes = max_payload_mb * 1024 * 1024

    def validate_predict_request(self, data):
        """
        Validate a /predict request payload.

        Returns:
            (True, None) on valid input.
            (False, error_message) on invalid input.
        """
        if data is None:
            return False, "Request body must be JSON"

        required = ["ciphertexts", "shape", "on_shape"]
        for field in required:
            if field not in data:
                return False, f"Missing required field: {field}"

        if not isinstance(data["ciphertexts"], list):
            return False, "'ciphertexts' must be a list"

        if len(data["ciphertexts"]) == 0:
            return False, "'ciphertexts' must not be empty"

        # Validate base64 encoding and total size
        total_size = 0
        for i, ct in enumerate(data["ciphertexts"]):
            if not isinstance(ct, str):
                return False, f"ciphertexts[{i}] must be a string"
            try:
                decoded = base64.b64decode(ct)
                total_size += len(decoded)
            except Exception:
                return False, f"ciphertexts[{i}] is not valid base64"

        if total_size > self.max_payload_bytes:
            return False, (f"Payload too large: {total_size / 1024 / 1024:.1f}MB "
                           f"(max {self.max_payload_bytes / 1024 / 1024:.0f}MB)")

        if not isinstance(data["shape"], list) or not all(
                isinstance(x, int) and x > 0 for x in data["shape"]):
            return False, "'shape' must be a list of positive integers"

        if not isinstance(data["on_shape"], list) or not all(
                isinstance(x, int) and x > 0 for x in data["on_shape"]):
            return False, "'on_shape' must be a list of positive integers"

        return True, None

    def validate_audio_upload(self, file_storage, max_mb=None):
        """
        Validate an uploaded audio file.

        Returns:
            (True, None) on valid file.
            (False, error_message) on invalid file.
        """
        if file_storage is None:
            return False, "No file provided"

        if not file_storage.filename:
            return False, "Empty filename"

        max_bytes = (max_mb or self.max_payload_bytes / 1024 / 1024) * 1024 * 1024

        # Check file extension
        allowed_ext = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        ext = "." + file_storage.filename.rsplit(".", 1)[-1].lower() \
            if "." in file_storage.filename else ""
        if ext not in allowed_ext:
            return False, f"Unsupported file type: {ext}"

        # Check file size by reading header
        file_storage.seek(0, 2)  # Seek to end
        size = file_storage.tell()
        file_storage.seek(0)  # Reset

        if size > max_bytes:
            return False, (f"File too large: {size / 1024 / 1024:.1f}MB "
                           f"(max {max_bytes / 1024 / 1024:.0f}MB)")

        if size == 0:
            return False, "Empty file"

        # Check magic bytes for WAV (RIFF header) or FLAC
        header = file_storage.read(12)
        file_storage.seek(0)

        if ext == ".wav" and header[:4] != b"RIFF":
            return False, "Invalid WAV file (bad RIFF header)"
        if ext == ".flac" and header[:4] != b"fLaC":
            return False, "Invalid FLAC file (bad header)"

        return True, None


# --- Key Management ---

def hash_key(key_bytes):
    """Generate a SHA-256 hash of key material for integrity checking."""
    return hashlib.sha256(key_bytes).hexdigest()


def encrypt_key_file(key_bytes, password):
    """
    Encrypt a key file with a password using PBKDF2 + XOR.

    Simple but effective for key-at-rest protection.
    For production HSM integration, replace this.
    """
    salt = hashlib.sha256(password.encode()).digest()[:16]
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000, dklen=len(key_bytes))
    encrypted = bytes(a ^ b for a, b in zip(key_bytes, dk))
    return salt + encrypted


def decrypt_key_file(encrypted_data, password):
    """Decrypt a key file encrypted with encrypt_key_file."""
    salt = encrypted_data[:16]
    encrypted = encrypted_data[16:]
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000, dklen=len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, dk))
