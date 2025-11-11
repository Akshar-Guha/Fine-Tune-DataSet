"""Metrics collection for model performance and system health."""

from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time


class MetricsCollector:
    """Collects and exposes metrics for monitoring."""

    def __init__(self):
        self.registry = CollectorRegistry()

        # Model performance metrics
        self.model_accuracy = Gauge(
            "model_accuracy", "Model accuracy", ["model_id"], registry=self.registry
        )
        self.model_latency = Histogram(
            "model_inference_latency_seconds",
            "Inference latency",
            ["model_id"],
            registry=self.registry,
        )
        self.inference_count = Counter(
            "model_inference_total",
            "Total inferences",
            ["model_id"],
            registry=self.registry,
        )

        # System metrics
        self.error_count = Counter(
            "system_errors_total",
            "Total errors",
            ["error_type"],
            registry=self.registry,
        )

    def record_inference(self, model_id: str, latency: float):
        """Record inference metrics."""
        self.model_latency.labels(model_id=model_id).observe(latency)
        self.inference_count.labels(model_id=model_id).inc()

    def update_accuracy(self, model_id: str, accuracy: float):
        """Update model accuracy."""
        self.model_accuracy.labels(model_id=model_id).set(accuracy)

    def record_error(self, error_type: str):
        """Record error."""
        self.error_count.labels(error_type=error_type).inc()


# Example
if __name__ == "__main__":
    mc = MetricsCollector()
    mc.record_inference("model_v1", 0.123)
    mc.update_accuracy("model_v1", 0.95)
    print("✅ Metrics collection test complete")
