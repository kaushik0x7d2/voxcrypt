"""
Reliability utilities for Orion Voice.

Provides retry with exponential backoff, circuit breaker,
and inference request queue.
"""

import time
import random
import queue
import threading
import logging
from concurrent.futures import Future
from functools import wraps

from speaker_verify.logging_config import get_logger

logger = get_logger("resilience")


# --- Retry Decorator ---

def retry(max_retries=3, backoff_base=1.0,
          retryable=(ConnectionError, TimeoutError)):
    """
    Retry decorator with exponential backoff and jitter.

    Args:
        max_retries: Maximum number of retry attempts.
        backoff_base: Base delay in seconds (doubled each retry).
        retryable: Tuple of exception types that trigger a retry.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = backoff_base * (2 ** attempt)
                        jitter = random.uniform(0, delay * 0.1)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for "
                            f"{func.__name__}: {e}. "
                            f"Waiting {delay + jitter:.1f}s")
                        time.sleep(delay + jitter)
            raise last_exc
        return wrapper
    return decorator


# --- Circuit Breaker ---

class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    States:
        CLOSED: Normal operation, requests pass through.
        OPEN: Failures exceeded threshold, requests fail fast.
        HALF_OPEN: Testing recovery, allows one request.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold=5, recovery_timeout=30,
                 name="default"):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._lock = threading.Lock()

    @property
    def state(self):
        with self._lock:
            if self._state == self.OPEN:
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' -> HALF_OPEN")
            return self._state

    def call(self, func, *args, **kwargs):
        """Execute a function through the circuit breaker."""
        state = self.state

        if state == self.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Retry after {self.recovery_timeout}s.")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            self._failure_count = 0
            if self._state == self.HALF_OPEN:
                self._state = self.CLOSED
                logger.info(f"Circuit breaker '{self.name}' -> CLOSED")

    def _on_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = self.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' -> OPEN "
                    f"after {self._failure_count} failures")


class CircuitBreakerOpenError(Exception):
    pass


# --- Inference Queue ---

class InferenceQueue:
    """
    Thread-safe queue that serializes FHE inference requests.

    FHE inference is single-threaded by nature (shared scheme state),
    so this queue ensures requests are processed one at a time while
    allowing the server to accept concurrent connections.
    """

    def __init__(self, max_queue_size=100, timeout=120):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._timeout = timeout
        self._running = False
        self._worker_thread = None

    def start(self):
        """Start the inference worker thread."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True)
        self._worker_thread.start()
        logger.info("Inference queue started")

    def stop(self):
        """Stop the inference worker and drain the queue."""
        self._running = False
        self._queue.put(None)  # Sentinel to unblock worker
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Inference queue stopped")

    def submit(self, inference_fn, *args, **kwargs):
        """
        Submit an inference request.

        Returns a Future that resolves to the inference result.
        Raises queue.Full if the queue is at capacity.
        """
        future = Future()
        try:
            self._queue.put(
                (future, inference_fn, args, kwargs),
                timeout=self._timeout)
        except queue.Full:
            future.set_exception(
                QueueFullError("Inference queue is full. Try again later."))
        return future

    def _worker(self):
        """Process inference requests one at a time."""
        while self._running:
            try:
                item = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                break

            future, fn, args, kwargs = item
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

    @property
    def pending(self):
        return self._queue.qsize()


class QueueFullError(Exception):
    pass
