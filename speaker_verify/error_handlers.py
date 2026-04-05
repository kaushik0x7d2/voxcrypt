"""
Centralized error handling for Orion Voice.

Custom exceptions and Flask error handlers that return consistent JSON responses.
"""

from flask import jsonify
from speaker_verify.logging_config import get_logger, get_request_id

logger = get_logger("errors")


# --- Custom Exceptions ---


class OrionVoiceError(Exception):
    """Base exception for Orion Voice."""

    status_code = 500
    error_code = "INTERNAL_ERROR"

    def __init__(self, message="Internal server error", details=None):
        super().__init__(message)
        self.message = message
        self.details = details


class ValidationError(OrionVoiceError):
    status_code = 400
    error_code = "VALIDATION_ERROR"


class AuthenticationError(OrionVoiceError):
    status_code = 401
    error_code = "AUTH_FAILED"


class RateLimitError(OrionVoiceError):
    status_code = 429
    error_code = "RATE_LIMITED"

    def __init__(self, message="Rate limit exceeded", retry_after=60):
        super().__init__(message)
        self.retry_after = retry_after


class PayloadTooLargeError(OrionVoiceError):
    status_code = 413
    error_code = "PAYLOAD_TOO_LARGE"


class FHEInferenceError(OrionVoiceError):
    status_code = 500
    error_code = "FHE_ERROR"


class ServiceUnavailableError(OrionVoiceError):
    status_code = 503
    error_code = "SERVICE_UNAVAILABLE"


# --- Error Response Builder ---


def error_response(error_code, message, status_code, details=None):
    """Build a consistent error response."""
    body = {
        "error": message,
        "code": error_code,
    }
    request_id = get_request_id()
    if request_id:
        body["request_id"] = request_id
    if details:
        body["details"] = details

    return jsonify(body), status_code


# --- Flask Error Handler Registration ---


def register_error_handlers(app):
    """Register error handlers on a Flask app."""

    @app.errorhandler(OrionVoiceError)
    def handle_orion_error(e):
        logger.error(
            f"{e.error_code}: {e.message}", extra={"extra_data": {"details": e.details}}
        )
        resp = error_response(e.error_code, e.message, e.status_code, e.details)
        if isinstance(e, RateLimitError):
            resp[0].headers["Retry-After"] = str(e.retry_after)
        return resp

    @app.errorhandler(400)
    def handle_400(e):
        return error_response("BAD_REQUEST", str(e), 400)

    @app.errorhandler(404)
    def handle_404(e):
        return error_response("NOT_FOUND", "Endpoint not found", 404)

    @app.errorhandler(405)
    def handle_405(e):
        return error_response("METHOD_NOT_ALLOWED", "Method not allowed", 405)

    @app.errorhandler(413)
    def handle_413(e):
        return error_response("PAYLOAD_TOO_LARGE", "Request too large", 413)

    @app.errorhandler(500)
    def handle_500(e):
        logger.exception("Unhandled exception")
        return error_response("INTERNAL_ERROR", "Internal server error", 500)
