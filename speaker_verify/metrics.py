"""
In-memory metrics collection for Orion Voice.

Thread-safe counters, gauges, and histograms.
Exposes a /metrics-compatible text endpoint.
"""

import threading
import time


class Counter:
    """Thread-safe monotonically increasing counter."""

    def __init__(self, name, description=""):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount=1):
        with self._lock:
            self._value += amount

    @property
    def value(self):
        return self._value


class Gauge:
    """Thread-safe gauge (can go up and down)."""

    def __init__(self, name, description=""):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()

    def set(self, value):
        with self._lock:
            self._value = value

    def inc(self, amount=1):
        with self._lock:
            self._value += amount

    def dec(self, amount=1):
        with self._lock:
            self._value -= amount

    @property
    def value(self):
        return self._value


class Histogram:
    """Thread-safe histogram with summary statistics."""

    def __init__(self, name, description=""):
        self.name = name
        self.description = description
        self._values = []
        self._lock = threading.Lock()

    def observe(self, value):
        with self._lock:
            self._values.append(value)

    @property
    def count(self):
        return len(self._values)

    @property
    def sum(self):
        return sum(self._values) if self._values else 0

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0

    @property
    def min(self):
        return min(self._values) if self._values else 0

    @property
    def max(self):
        return max(self._values) if self._values else 0


class MetricsRegistry:
    """Central registry for all application metrics."""

    def __init__(self):
        self.start_time = time.time()

        # Inference metrics
        self.inference_total = Counter("inference_total", "Total inference requests")
        self.inference_errors = Counter("inference_errors", "Total inference errors")
        self.inference_duration = Histogram(
            "inference_duration_seconds", "Inference duration in seconds"
        )

        # Request metrics
        self.requests_total = Counter("requests_total", "Total HTTP requests")
        self.active_requests = Gauge("active_requests", "Currently active requests")
        self.auth_failures = Counter("auth_failures", "Authentication failures")
        self.rate_limit_hits = Counter("rate_limit_hits", "Rate limit rejections")

        # FHE metrics
        self.ciphertext_size = Histogram(
            "ciphertext_size_bytes", "Ciphertext payload size"
        )
        self.precision_bits = Histogram("precision_bits", "FHE precision in bits")

    @property
    def uptime(self):
        return time.time() - self.start_time

    def to_dict(self):
        """Export all metrics as a dict."""
        return {
            "uptime_seconds": round(self.uptime, 1),
            "inference": {
                "total": self.inference_total.value,
                "errors": self.inference_errors.value,
                "avg_duration": round(self.inference_duration.avg, 3),
                "min_duration": round(self.inference_duration.min, 3),
                "max_duration": round(self.inference_duration.max, 3),
            },
            "requests": {
                "total": self.requests_total.value,
                "active": self.active_requests.value,
                "auth_failures": self.auth_failures.value,
                "rate_limit_hits": self.rate_limit_hits.value,
            },
        }

    def to_text(self):
        """Export metrics in Prometheus-compatible text format."""
        lines = []
        lines.append("# HELP uptime_seconds Server uptime")
        lines.append(f"uptime_seconds {self.uptime:.1f}")
        lines.append("# HELP inference_total Total inferences")
        lines.append(f"inference_total {self.inference_total.value}")
        lines.append(f"inference_errors {self.inference_errors.value}")
        lines.append(f"inference_duration_avg {self.inference_duration.avg:.3f}")
        lines.append(f"requests_total {self.requests_total.value}")
        lines.append(f"active_requests {self.active_requests.value}")
        lines.append(f"auth_failures {self.auth_failures.value}")
        lines.append(f"rate_limit_hits {self.rate_limit_hits.value}")
        return "\n".join(lines)


# Global registry
registry = MetricsRegistry()
