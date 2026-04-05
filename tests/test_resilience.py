"""Tests for reliability utilities."""

import time
import pytest
from speaker_verify.resilience import (
    retry, CircuitBreaker, CircuitBreakerOpenError,
    InferenceQueue, QueueFullError)


class TestRetry:
    def test_succeeds_first_try(self):
        call_count = 0

        @retry(max_retries=3, backoff_base=0.01)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count == 1

    def test_succeeds_after_retries(self):
        call_count = 0

        @retry(max_retries=3, backoff_base=0.01,
               retryable=(ConnectionError,))
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "ok"

        result = fail_twice()
        assert result == "ok"
        assert call_count == 3

    def test_exhausts_retries(self):
        @retry(max_retries=2, backoff_base=0.01,
               retryable=(ConnectionError,))
        def always_fail():
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            always_fail()

    def test_non_retryable_exception(self):
        @retry(max_retries=3, backoff_base=0.01,
               retryable=(ConnectionError,))
        def raise_value_error():
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            raise_value_error()


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitBreaker.CLOSED

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=100)
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
            except ValueError:
                pass

        assert cb.state == CircuitBreaker.OPEN
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(lambda: "ok")

    def test_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass

        cb.call(lambda: "ok")
        assert cb._failure_count == 0

    def test_half_open_recovery(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass

        assert cb.state == CircuitBreaker.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitBreaker.HALF_OPEN

        cb.call(lambda: "ok")
        assert cb.state == CircuitBreaker.CLOSED


class TestInferenceQueue:
    def test_submit_and_get_result(self):
        q = InferenceQueue(max_queue_size=10)
        q.start()
        try:
            future = q.submit(lambda: 42)
            result = future.result(timeout=5)
            assert result == 42
        finally:
            q.stop()

    def test_exception_propagation(self):
        q = InferenceQueue(max_queue_size=10)
        q.start()
        try:
            future = q.submit(lambda: (_ for _ in ()).throw(
                ValueError("test error")))
            with pytest.raises(ValueError, match="test error"):
                future.result(timeout=5)
        finally:
            q.stop()

    def test_ordering(self):
        q = InferenceQueue(max_queue_size=10)
        q.start()
        try:
            results = []
            futures = [q.submit(lambda i=i: results.append(i) or i)
                       for i in range(5)]
            for f in futures:
                f.result(timeout=5)
            assert results == [0, 1, 2, 3, 4]
        finally:
            q.stop()
