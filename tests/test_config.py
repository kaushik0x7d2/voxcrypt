"""Tests for configuration management."""

import os
import pytest
from speaker_verify.config import Config


class TestConfig:
    def test_defaults(self):
        config = Config()
        assert config.server.port == 5000
        assert config.server.host == "127.0.0.1"
        assert config.security.require_auth is False
        assert config.logging.level == "INFO"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("ORION_VOICE_PORT", "8080")
        monkeypatch.setenv("ORION_VOICE_REQUIRE_AUTH", "true")
        monkeypatch.setenv("ORION_VOICE_API_KEY", "test-key")
        monkeypatch.setenv("ORION_VOICE_LOG_LEVEL", "DEBUG")

        config = Config.from_env()
        assert config.server.port == 8080
        assert config.security.require_auth is True
        assert config.security.api_key == "test-key"
        assert config.logging.level == "DEBUG"

    def test_validate_valid(self):
        config = Config()
        config.validate()  # Should not raise

    def test_validate_invalid_port(self):
        config = Config()
        config.server.port = 99999
        with pytest.raises(ValueError, match="Invalid port"):
            config.validate()

    def test_validate_auth_without_key(self):
        config = Config()
        config.security.require_auth = True
        config.security.api_key = ""
        with pytest.raises(ValueError, match="API_KEY required"):
            config.validate()

    def test_validate_invalid_log_level(self):
        config = Config()
        config.logging.level = "VERBOSE"
        with pytest.raises(ValueError, match="Invalid log level"):
            config.validate()

    def test_production_flag(self, monkeypatch):
        monkeypatch.setenv("ORION_VOICE_PRODUCTION", "true")
        config = Config.from_env()
        assert config.server.production is True

    def test_production_flag_false(self, monkeypatch):
        monkeypatch.setenv("ORION_VOICE_PRODUCTION", "false")
        config = Config.from_env()
        assert config.server.production is False
