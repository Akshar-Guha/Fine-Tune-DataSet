"""
Data Quality Validation Framework.

Validates completeness, uniqueness, validity, and consistency of data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from scipy import stats


class DataQualityValidator:
    """Comprehensive data quality validation."""

    def __init__(self):
        """Initialize Data Quality Validator."""
        self.validation_results = []

    def validate_dataset(
        self, df: pd.DataFrame, rules: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Validate a dataset against quality rules.

        Args:
            df: DataFrame to validate
            rules: Custom validation rules

        Returns:
            Quality report with score and issues
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "checks": {},
            "issues": [],
            "quality_score": 0.0,
        }

        # Run validation checks
        results["checks"]["completeness"] = self._check_completeness(df)
        results["checks"]["uniqueness"] = self._check_uniqueness(df)
        results["checks"]["validity"] = self._check_validity(df, rules)
        results["checks"]["consistency"] = self._check_consistency(df)
        results["checks"]["outliers"] = self._detect_outliers(df)

        # Calculate quality score (0-100)
        results["quality_score"] = self._calculate_quality_score(results["checks"])

        # Collect all issues
        for check_name, check_result in results["checks"].items():
            if "issues" in check_result:
                results["issues"].extend(check_result["issues"])

        return results

    def _check_completeness(self, df: pd.DataFrame) -> Dict:
        """Check for missing values."""
        missing_stats = df.isnull().sum()
        missing_pct = (missing_stats / len(df) * 100).round(2)

        issues = []
        for col, pct in missing_pct.items():
            if pct > 5:  # More than 5% missing
                issues.append(
                    {
                        "column": col,
                        "type": "completeness",
                        "severity": "high" if pct > 20 else "medium",
                        "message": f"{col} has {pct}% missing values",
                    }
                )

        return {
            "status": "pass" if len(issues) == 0 else "fail",
            "missing_count": int(missing_stats.sum()),
            "missing_by_column": missing_stats.to_dict(),
            "issues": issues,
        }

    def _check_uniqueness(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate rows."""
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df) * 100).round(2)

        issues = []
        if duplicate_pct > 1:  # More than 1% duplicates
            issues.append(
                {
                    "type": "uniqueness",
                    "severity": "medium",
                    "message": f"{duplicate_pct}% duplicate rows found",
                }
            )

        return {
            "status": "pass" if duplicate_pct < 1 else "fail",
            "duplicate_count": int(duplicate_count),
            "duplicate_percentage": float(duplicate_pct),
            "issues": issues,
        }

    def _check_validity(
        self, df: pd.DataFrame, rules: Optional[Dict] = None
    ) -> Dict:
        """Check data validity against rules."""
        issues = []

        # Check data types
        for col in df.columns:
            if df[col].dtype == "object":
                # Check for valid strings
                invalid = df[col].str.len() == 0
                invalid_count = invalid.sum()
                if invalid_count > 0:
                    issues.append(
                        {
                            "column": col,
                            "type": "validity",
                            "severity": "low",
                            "message": f"{col} has {invalid_count} empty strings",
                        }
                    )

        # Apply custom rules
        if rules:
            for col, rule in rules.items():
                if col not in df.columns:
                    continue

                if "min" in rule:
                    invalid = df[col] < rule["min"]
                    if invalid.any():
                        issues.append(
                            {
                                "column": col,
                                "type": "validity",
                                "severity": "high",
                                "message": f"{col} has values below min {rule['min']}",
                            }
                        )

                if "max" in rule:
                    invalid = df[col] > rule["max"]
                    if invalid.any():
                        issues.append(
                            {
                                "column": col,
                                "type": "validity",
                                "severity": "high",
                                "message": f"{col} has values above max {rule['max']}",
                            }
                        )

        return {
            "status": "pass" if len(issues) == 0 else "fail",
            "issues": issues,
        }

    def _check_consistency(self, df: pd.DataFrame) -> Dict:
        """Check data consistency."""
        issues = []

        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                issues.append(
                    {
                        "column": col,
                        "type": "consistency",
                        "severity": "low",
                        "message": f"{col} has only one unique value",
                    }
                )

        return {
            "status": "pass" if len(issues) == 0 else "warning",
            "issues": issues,
        }

    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers using IQR method."""
        issues = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_pct = (outliers / len(df) * 100).round(2)

            if outlier_pct > 5:  # More than 5% outliers
                issues.append(
                    {
                        "column": col,
                        "type": "outliers",
                        "severity": "medium",
                        "message": f"{col} has {outlier_pct}% outliers",
                    }
                )

        return {
            "status": "pass" if len(issues) == 0 else "warning",
            "issues": issues,
        }

    def _calculate_quality_score(self, checks: Dict) -> float:
        """Calculate overall quality score (0-100)."""
        scores = []

        # Completeness (30% weight)
        completeness_check = checks.get("completeness", {})
        missing_count = completeness_check.get("missing_count", 0)
        if missing_count == 0:
            scores.append(30.0)
        else:
            scores.append(max(0, 30.0 - (missing_count / 100)))

        # Uniqueness (25% weight)
        uniqueness_check = checks.get("uniqueness", {})
        duplicate_pct = uniqueness_check.get("duplicate_percentage", 0)
        scores.append(max(0, 25.0 - duplicate_pct))

        # Validity (25% weight)
        validity_check = checks.get("validity", {})
        validity_issues = len(validity_check.get("issues", []))
        scores.append(max(0, 25.0 - validity_issues))

        # Consistency (20% weight)
        consistency_check = checks.get("consistency", {})
        consistency_issues = len(consistency_check.get("issues", []))
        scores.append(max(0, 20.0 - consistency_issues))

        return round(sum(scores), 2)

    def generate_report(self, validation_results: Dict) -> str:
        """Generate human-readable quality report."""
        report = ["=" * 60]
        report.append("DATA QUALITY REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {validation_results['timestamp']}")
        report.append(f"Total Rows: {validation_results['total_rows']:,}")
        report.append(f"Total Columns: {validation_results['total_columns']}")
        report.append(f"Quality Score: {validation_results['quality_score']}/100")
        report.append("")

        # Issues summary
        issues = validation_results.get("issues", [])
        if issues:
            report.append(f"⚠️ Found {len(issues)} issues:")
            for issue in issues:
                severity = issue.get("severity", "unknown")
                message = issue.get("message", "")
                report.append(f"  [{severity.upper()}] {message}")
        else:
            report.append("✅ No issues found")

        report.append("=" * 60)
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Create sample dataset
    df = pd.DataFrame(
        {
            "id": range(100),
            "name": ["User_" + str(i) for i in range(100)],
            "age": np.random.randint(18, 80, 100),
            "score": np.random.uniform(0, 100, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
        }
    )

    # Add some quality issues
    df.loc[5:10, "name"] = None  # Missing values
    df.loc[90:95, "age"] = -1  # Invalid values

    # Validate
    validator = DataQualityValidator()

    # Define rules
    rules = {"age": {"min": 0, "max": 120}, "score": {"min": 0, "max": 100}}

    results = validator.validate_dataset(df, rules=rules)

    # Print report
    print(validator.generate_report(results))

    print(f"\n✅ Data quality validation test complete")
    print(f"Quality Score: {results['quality_score']}/100")
