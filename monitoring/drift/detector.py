"""Drift detection using statistical tests."""

import numpy as np
from scipy import stats
from typing import Dict


class DriftDetector:
    """Detects data, concept, and model drift."""

    def detect_data_drift(
        self, baseline: np.ndarray, current: np.ndarray, threshold: float = 0.05
    ) -> Dict:
        """Detect data drift using KS test."""
        statistic, p_value = stats.ks_2samp(baseline, current)

        return {
            "drift_detected": p_value < threshold,
            "p_value": float(p_value),
            "statistic": float(statistic),
            "method": "ks_test",
        }

    def detect_concept_drift(
        self, baseline_acc: float, current_acc: float, threshold: float = 0.05
    ) -> Dict:
        """Detect concept drift via accuracy degradation."""
        accuracy_drop = baseline_acc - current_acc
        drift_detected = accuracy_drop > threshold

        return {
            "drift_detected": drift_detected,
            "accuracy_drop": float(accuracy_drop),
            "threshold": threshold,
        }


# Example
if __name__ == "__main__":
    dd = DriftDetector()

    # Data drift test
    baseline = np.random.normal(0, 1, 1000)
    current = np.random.normal(0.5, 1, 1000)  # Shifted distribution
    result = dd.detect_data_drift(baseline, current)
    print(f"Data Drift: {result}")

    # Concept drift test
    result = dd.detect_concept_drift(0.95, 0.85)
    print(f"Concept Drift: {result}")

    print("✅ Drift detection test complete")
