"""Tests for security middleware."""

import pytest
from unittest.mock import MagicMock
from speaker_verify.security import (
    APIKeyAuth, RateLimiter, InputValidator,
    encrypt_key_file, decrypt_key_file, hash_key)


class TestAPIKeyAuth:
    def test_no_auth_required(self):
        auth = APIKeyAuth("secret", require=False)
        req = MagicMock()
        ok, err = auth.authenticate(req)
        assert ok is True
        assert err is None

    def test_missing_header(self):
        auth = APIKeyAuth("secret", require=True)
        req = MagicMock()
        req.headers = {}
        req.remote_addr = "127.0.0.1"
        ok, err = auth.authenticate(req)
        assert ok is False
        assert "Missing" in err

    def test_wrong_key(self):
        auth = APIKeyAuth("correct-key", require=True)
        req = MagicMock()
        req.headers = {"Authorization": "Bearer wrong-key"}
        req.remote_addr = "127.0.0.1"
        ok, err = auth.authenticate(req)
        assert ok is False
        assert "Invalid" in err

    def test_correct_key(self):
        auth = APIKeyAuth("my-secret", require=True)
        req = MagicMock()
        req.headers = {"Authorization": "Bearer my-secret"}
        ok, err = auth.authenticate(req)
        assert ok is True

    def test_no_bearer_prefix(self):
        auth = APIKeyAuth("key", require=True)
        req = MagicMock()
        req.headers = {"Authorization": "key"}
        req.remote_addr = "127.0.0.1"
        ok, err = auth.authenticate(req)
        assert ok is False


class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            ok, _ = rl.allow("1.2.3.4")
            assert ok is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.allow("1.2.3.4")
        ok, retry_after = rl.allow("1.2.3.4")
        assert ok is False
        assert retry_after > 0

    def test_different_ips_independent(self):
        rl = RateLimiter(max_requests=1, window_seconds=60)
        ok1, _ = rl.allow("1.1.1.1")
        ok2, _ = rl.allow("2.2.2.2")
        assert ok1 is True
        assert ok2 is True


class TestInputValidator:
    def test_valid_predict_request(self):
        v = InputValidator()
        data = {
            "ciphertexts": ["dGVzdA=="],
            "shape": [1, 40],
            "on_shape": [1, 40],
        }
        ok, err = v.validate_predict_request(data)
        assert ok is True

    def test_missing_field(self):
        v = InputValidator()
        ok, err = v.validate_predict_request({"ciphertexts": ["dGVzdA=="]})
        assert ok is False
        assert "shape" in err

    def test_null_body(self):
        v = InputValidator()
        ok, err = v.validate_predict_request(None)
        assert ok is False

    def test_empty_ciphertexts(self):
        v = InputValidator()
        data = {"ciphertexts": [], "shape": [1], "on_shape": [1]}
        ok, err = v.validate_predict_request(data)
        assert ok is False

    def test_invalid_base64(self):
        v = InputValidator()
        data = {"ciphertexts": ["not-base64!!!"],
                "shape": [1], "on_shape": [1]}
        ok, err = v.validate_predict_request(data)
        assert ok is False

    def test_invalid_shape(self):
        v = InputValidator()
        data = {"ciphertexts": ["dGVzdA=="],
                "shape": [-1], "on_shape": [1]}
        ok, err = v.validate_predict_request(data)
        assert ok is False


class TestKeyManagement:
    def test_encrypt_decrypt_roundtrip(self):
        key_bytes = os.urandom(256)
        encrypted = encrypt_key_file(key_bytes, "my-password")
        decrypted = decrypt_key_file(encrypted, "my-password")
        assert decrypted == key_bytes

    def test_wrong_password_differs(self):
        key_bytes = b"secret-key-data-here-1234567890ab"
        encrypted = encrypt_key_file(key_bytes, "correct")
        decrypted = decrypt_key_file(encrypted, "wrong")
        assert decrypted != key_bytes

    def test_hash_key_deterministic(self):
        key = b"test-key"
        assert hash_key(key) == hash_key(key)

    def test_hash_key_different_inputs(self):
        assert hash_key(b"key1") != hash_key(b"key2")


import os
