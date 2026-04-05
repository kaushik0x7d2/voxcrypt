"""
Centralized configuration for Orion Voice.

Loads from environment variables with sensible defaults.
All configuration is validated at startup.
"""

import os
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    workers: int = 4
    request_timeout: int = 120
    max_payload_mb: int = 50
    cors_origins: str = ""
    production: bool = False


@dataclass
class SecurityConfig:
    api_key: str = ""
    require_auth: bool = False
    tls_cert: str = ""
    tls_key: str = ""
    hmac_key: str = ""
    require_hmac: bool = False
    key_password: str = ""
    rate_limit: int = 10
    rate_limit_window: int = 60
    max_upload_mb: int = 50


@dataclass
class FHEConfig:
    config_path: str = ""
    decision_threshold: float = 0.5


@dataclass
class MLConfig:
    model_path: str = ""
    scaler_path: str = ""
    samples_path: str = ""
    data_root: str = ""


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"
    log_file: str = ""


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    fhe: FHEConfig = field(default_factory=FHEConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        config = cls()

        # Server
        config.server.host = os.environ.get("ORION_VOICE_HOST", "127.0.0.1")
        config.server.port = int(os.environ.get("ORION_VOICE_PORT", "5000"))
        config.server.workers = int(os.environ.get("ORION_VOICE_WORKERS", "4"))
        config.server.request_timeout = int(
            os.environ.get("ORION_VOICE_REQUEST_TIMEOUT", "120")
        )
        config.server.max_payload_mb = int(
            os.environ.get("ORION_VOICE_MAX_PAYLOAD_MB", "50")
        )
        config.server.cors_origins = os.environ.get("ORION_VOICE_CORS_ORIGINS", "")
        config.server.production = os.environ.get(
            "ORION_VOICE_PRODUCTION", ""
        ).lower() in ("true", "1", "yes")

        # Security
        config.security.api_key = os.environ.get("ORION_VOICE_API_KEY", "")
        config.security.require_auth = os.environ.get(
            "ORION_VOICE_REQUIRE_AUTH", ""
        ).lower() in ("true", "1", "yes")
        config.security.tls_cert = os.environ.get("ORION_VOICE_TLS_CERT", "")
        config.security.tls_key = os.environ.get("ORION_VOICE_TLS_KEY", "")
        config.security.rate_limit = int(os.environ.get("ORION_VOICE_RATE_LIMIT", "10"))
        config.security.rate_limit_window = int(
            os.environ.get("ORION_VOICE_RATE_LIMIT_WINDOW", "60")
        )
        config.security.max_upload_mb = int(
            os.environ.get("ORION_VOICE_MAX_UPLOAD_MB", "50")
        )
        config.security.key_password = os.environ.get("ORION_VOICE_KEY_PASSWORD", "")
        config.security.require_hmac = os.environ.get(
            "ORION_VOICE_REQUIRE_HMAC", ""
        ).lower() in ("true", "1", "yes")

        # FHE
        config.fhe.config_path = os.environ.get("ORION_VOICE_FHE_CONFIG_PATH", "")
        config.fhe.decision_threshold = float(
            os.environ.get("ORION_VOICE_DECISION_THRESHOLD", "0.5")
        )

        # ML
        config.ml.model_path = os.environ.get("ORION_VOICE_MODEL_PATH", "")
        config.ml.scaler_path = os.environ.get("ORION_VOICE_SCALER_PATH", "")
        config.ml.samples_path = os.environ.get("ORION_VOICE_SAMPLES_PATH", "")
        config.ml.data_root = os.environ.get("ORION_VOICE_DATA_ROOT", "")

        # Logging
        config.logging.level = os.environ.get("ORION_VOICE_LOG_LEVEL", "INFO")
        config.logging.format = os.environ.get("ORION_VOICE_LOG_FORMAT", "json")
        config.logging.log_file = os.environ.get("ORION_VOICE_LOG_FILE", "")

        return config

    def validate(self):
        """Validate configuration. Raises ValueError on invalid config."""
        errors = []

        if not 1 <= self.server.port <= 65535:
            errors.append(f"Invalid port: {self.server.port}")

        if self.server.workers < 1:
            errors.append(f"Workers must be >= 1, got {self.server.workers}")

        if self.security.require_auth and not self.security.api_key:
            errors.append("ORION_VOICE_API_KEY required when auth is enabled")

        if self.security.tls_cert and not os.path.exists(self.security.tls_cert):
            errors.append(f"TLS cert not found: {self.security.tls_cert}")

        if self.security.tls_key and not os.path.exists(self.security.tls_key):
            errors.append(f"TLS key not found: {self.security.tls_key}")

        if self.logging.level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append(f"Invalid log level: {self.logging.level}")

        if errors:
            raise ValueError("Configuration errors:\n  " + "\n  ".join(errors))
