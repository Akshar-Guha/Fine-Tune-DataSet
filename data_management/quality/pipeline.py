"""Automated dataset ingestion and validation pipeline."""
from __future__ import annotations

import json
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Tuple

import pandas as pd

from data_management.quality.pipeline_config import (
    AutoLabelingRule,
    DatasetPipelineConfig,
    load_pipeline_config,
)
from data_management.quality.validator import DataQualityValidator
from services.dataset_registry import DatasetRegistry


@dataclass
class DatasetPipelineResult:
    """Result of running the dataset ingestion pipeline."""

    dataset_id: str
    dataset_name: str
    raw_file_path: Path
    processed_file_path: Optional[Path]
    registry_path: Optional[Path]
    quality_report_path: Path
    rows: int
    columns: int
    text_column: str
    label_column: str
    quality_report: Dict[str, Any]
    quality_passed: bool

    @property
    def quality_score(self) -> float:
        return float(self.quality_report.get("quality_score", 0.0))

    @property
    def issues(self) -> List[Dict[str, Any]]:
        return list(self.quality_report.get("issues", []))


class DatasetIngestionPipeline:
    """Clean, label, validate, and register uploaded datasets."""

    def __init__(
        self,
        config: Optional[DatasetPipelineConfig] = None,
        registry: Optional[DatasetRegistry] = None,
    ) -> None:
        self.config = config or load_pipeline_config()
        self.registry = registry or DatasetRegistry(
            datasets_dir=self.config.dataset_dir
        )
        self.validator = DataQualityValidator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        dataset_name: Optional[str],
        upload_filename: str,
        file_obj: IO[bytes],
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        dataset_id: Optional[str] = None,
        enable_auto_labeling: Optional[bool] = None,
        auto_labeling_rules: Optional[List[AutoLabelingRule]] = None,
        quality_score_threshold: Optional[int] = None,
        strip_text: Optional[bool] = None,
        drop_missing_text: Optional[bool] = None,
        drop_missing_label: Optional[bool] = None,
        lowercase_labels: Optional[bool] = None,
        drop_duplicates: Optional[bool] = None,
    ) -> DatasetPipelineResult:
        """Execute ingestion pipeline for an uploaded file."""

        dataset_id = dataset_id or self.generate_dataset_id(
            dataset_name, upload_filename
        )
        raw_path = self._persist_upload(dataset_id, upload_filename, file_obj)
        normalized_rules = self._normalize_auto_labeling_rules(auto_labeling_rules)
        overrides = self._prepare_overrides(
            enable_auto_labeling=enable_auto_labeling,
            auto_labeling_rules=normalized_rules,
            quality_score_threshold=quality_score_threshold,
            strip_text=strip_text,
            drop_missing_text=drop_missing_text,
            drop_missing_label=drop_missing_label,
            lowercase_labels=lowercase_labels,
            drop_duplicates=drop_duplicates,
        )

        with self._temporary_config(overrides):
            return self._execute_pipeline(
                raw_path=raw_path,
                dataset_id=dataset_id,
                dataset_name=dataset_name or dataset_id,
                text_column=text_column,
                label_column=label_column,
            )

    def run_from_local_path(
        self,
        *,
        local_path: str,
        dataset_name: str,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        auto_labeling_rules: Optional[List[AutoLabelingRule]] = None,
        enable_auto_labeling: Optional[bool] = None,
        quality_score_threshold: Optional[int] = None,
        strip_text: Optional[bool] = None,
        drop_missing_text: Optional[bool] = None,
        drop_missing_label: Optional[bool] = None,
        lowercase_labels: Optional[bool] = None,
        dataset_id: Optional[str] = None,
        drop_duplicates: Optional[bool] = None,
    ) -> DatasetPipelineResult:
        """Execute ingestion pipeline for an existing local file."""

        dataset_id = dataset_id or self.generate_dataset_id(
            dataset_name, local_path
        )
        file_path = Path(local_path)

        # Create a temporary copy in uploads directory for processing
        raw_path = self._persist_local_file(dataset_id, file_path)
        normalized_rules = self._normalize_auto_labeling_rules(auto_labeling_rules)
        overrides = self._prepare_overrides(
            enable_auto_labeling=enable_auto_labeling,
            auto_labeling_rules=normalized_rules,
            quality_score_threshold=quality_score_threshold,
            strip_text=strip_text,
            drop_missing_text=drop_missing_text,
            drop_missing_label=drop_missing_label,
            lowercase_labels=lowercase_labels,
            drop_duplicates=drop_duplicates,
        )

        with self._temporary_config(overrides):
            return self._execute_pipeline(
                raw_path=raw_path,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                text_column=text_column,
                label_column=label_column,
            )

    def run_from_huggingface_dataset(
        self,
        *,
        dataset_path: str,
        dataset_name: str,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        auto_labeling_rules: Optional[List[AutoLabelingRule]] = None,
        enable_auto_labeling: Optional[bool] = None,
        quality_score_threshold: Optional[int] = None,
        strip_text: Optional[bool] = None,
        drop_missing_text: Optional[bool] = None,
        drop_missing_label: Optional[bool] = None,
        lowercase_labels: Optional[bool] = None,
        dataset_id: Optional[str] = None,
        drop_duplicates: Optional[bool] = None,
    ) -> DatasetPipelineResult:
        """Execute ingestion pipeline for an already-downloaded HuggingFace dataset."""

        dataset_id = dataset_id or self.generate_dataset_id(
            dataset_name, dataset_path
        )

        # Use the dataset path directly - no need to copy since it's already a HF dataset
        raw_path = Path(dataset_path)

        normalized_rules = self._normalize_auto_labeling_rules(auto_labeling_rules)
        overrides = self._prepare_overrides(
            enable_auto_labeling=enable_auto_labeling,
            auto_labeling_rules=normalized_rules,
            quality_score_threshold=quality_score_threshold,
            strip_text=strip_text,
            drop_missing_text=drop_missing_text,
            drop_missing_label=drop_missing_label,
            lowercase_labels=lowercase_labels,
            drop_duplicates=drop_duplicates,
        )

        with self._temporary_config(overrides):
            return self._execute_pipeline_from_hf_dataset(
                dataset_path=raw_path,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                text_column=text_column,
                label_column=label_column,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def generate_dataset_id(
        self, dataset_name: Optional[str], upload_filename: str
    ) -> str:
        """Generate a stable dataset identifier for an upload."""

        return self._build_dataset_id(dataset_name or upload_filename)

    def _build_dataset_id(self, base_name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", base_name.lower()).strip("-")
        if not slug:
            slug = "dataset"

        candidate = slug
        counter = 1
        while (
            self.config.processed_path / candidate
        ).exists() or (
            self.config.uploads_path / candidate
        ).exists():
            counter += 1
            candidate = f"{slug}-{counter}"
        return candidate

    def _persist_upload(
        self,
        dataset_id: str,
        upload_filename: str,
        file_obj: IO[bytes],
    ) -> Path:
        """Persist an uploaded file into the staging uploads directory."""

        destination_dir = self.config.uploads_path / dataset_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / upload_filename

        file_obj.seek(0)
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file_obj, buffer)

        return destination

    def _persist_local_file(self, dataset_id: str, file_path: Path) -> Path:
        """Copy a local file to the uploads directory for processing."""
        destination_dir = self.config.uploads_path / dataset_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / file_path.name

        shutil.copy2(file_path, destination)
        return destination

    @contextmanager
    def _temporary_config(self, overrides: Dict[str, Any]):
        """Temporarily apply configuration overrides while running the pipeline."""

        if not overrides:
            yield self.config
            return

        original_config = self.config
        self.config = replace(self.config, **overrides)
        try:
            yield self.config
        finally:
            self.config = original_config

    def _prepare_overrides(
        self,
        *,
        enable_auto_labeling: Optional[bool],
        auto_labeling_rules: Optional[List[AutoLabelingRule]],
        quality_score_threshold: Optional[int],
        strip_text: Optional[bool],
        drop_missing_text: Optional[bool],
        drop_missing_label: Optional[bool],
        lowercase_labels: Optional[bool],
        drop_duplicates: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Collect non-null overrides for temporary configuration."""

        overrides: Dict[str, Any] = {}
        if auto_labeling_rules is not None:
            overrides["auto_labeling_rules"] = auto_labeling_rules
        if enable_auto_labeling is not None:
            overrides["enable_auto_labeling"] = enable_auto_labeling
        if quality_score_threshold is not None:
            overrides["quality_score_threshold"] = float(
            quality_score_threshold)
        if strip_text is not None:
            overrides["strip_text"] = strip_text
        if drop_missing_text is not None:
            overrides["drop_missing_text"] = drop_missing_text
        if drop_missing_label is not None:
            overrides["drop_missing_label"] = drop_missing_label
        if lowercase_labels is not None:
            overrides["lowercase_labels"] = lowercase_labels
        if drop_duplicates is not None:
            overrides["drop_duplicates"] = drop_duplicates

        if (
            enable_auto_labeling is None
            and auto_labeling_rules is not None
        ):
            overrides.setdefault("enable_auto_labeling", bool(auto_labeling_rules))

        return overrides

    def _normalize_auto_labeling_rules(
        self, rules: Optional[List[AutoLabelingRule]]
    ) -> Optional[List[AutoLabelingRule]]:
        """Ensure auto labeling rules are instances of AutoLabelingRule."""

        if rules is None:
            return None

        normalized: List[AutoLabelingRule] = []
        for rule in rules:
            if isinstance(rule, AutoLabelingRule):
                normalized.append(rule)
            else:
                # Assume rule is a dict-like object
                label = getattr(rule, "label", rule.get("label", "")) if hasattr(rule, "get") else getattr(rule, "label", "")
                keywords = getattr(rule, "keywords", rule.get("keywords", [])) if hasattr(rule, "get") else getattr(rule, "keywords", [])
                normalized.append(
                    AutoLabelingRule(
                        label=label,
                        keywords=list(keywords),
                    )
                )
        return normalized

    def _execute_pipeline(
        self,
        *,
        raw_path: Path,
        dataset_id: str,
        dataset_name: str,
        text_column: Optional[str],
        label_column: Optional[str],
    ) -> DatasetPipelineResult:
        """Process persisted raw data through transformation and validation."""

        dataframe = self._load_dataframe(raw_path)
        processed_df, resolved_columns = self._transform_dataframe(
            dataframe, text_column=text_column, label_column=label_column
        )

        quality_report = self._validate(processed_df)
        quality_passed = self._quality_passed(quality_report)

        processed_file_path: Optional[Path] = None
        registry_path: Optional[Path] = None

        if quality_passed:
            (
                processed_file_path,
                registry_path,
                report_path,
            ) = self._persist_outputs(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                dataframe=processed_df,
                quality_report=quality_report,
                text_column=resolved_columns[0],
                label_column=resolved_columns[1],
            )
        else:
            report_path = self._write_quality_report(
                directory=self.config.uploads_path / dataset_id,
                report=quality_report,
            )

        return DatasetPipelineResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            raw_file_path=raw_path,
            processed_file_path=processed_file_path,
            registry_path=registry_path,
            quality_report_path=report_path,
            rows=len(processed_df),
            columns=len(processed_df.columns),
            text_column=resolved_columns[0],
            label_column=resolved_columns[1],
            quality_report=quality_report,
            quality_passed=quality_passed,
        )

    def _load_dataframe(self, file_path: Path) -> pd.DataFrame:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(file_path)
        if suffix in {".json", ".jsonl"}:
            return pd.read_json(file_path, lines=suffix == ".jsonl")
        if suffix == ".parquet":
            return pd.read_parquet(file_path)

        raise ValueError(f"Unsupported file type: {suffix}")

    def _transform_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        text_column: Optional[str],
        label_column: Optional[str],
    ) -> Tuple[pd.DataFrame, Tuple[str, str]]:
        df = dataframe.copy()

        resolved_text_col = self._resolve_text_column(df, text_column)
        resolved_label_col = self._resolve_label_column(df, label_column)

        df[resolved_text_col] = df[resolved_text_col].astype(str)
        if self.config.strip_text:
            df[resolved_text_col] = df[resolved_text_col].str.strip()

        if self.config.drop_missing_text:
            df = df[df[resolved_text_col].notnull()]
            df = df[df[resolved_text_col].str.len() > 0]

        # Prepare label column
        if resolved_label_col not in df.columns:
            df[resolved_label_col] = None

        # Normalise labels
        if self.config.lowercase_labels:
            df[resolved_label_col] = df[resolved_label_col].astype(str).str.lower()

        df.loc[
            df[resolved_label_col].isin({"none", "nan", ""}),
            resolved_label_col,
        ] = None

        if self.config.allowed_label_values:
            allowed = {value.lower() for value in self.config.allowed_label_values}
            df.loc[
                ~df[resolved_label_col].isin(allowed),
                resolved_label_col,
            ] = None

        if self.config.enable_auto_labeling:
            df = self._apply_auto_labeling(df, resolved_text_col, resolved_label_col)

        if self.config.drop_missing_label:
            df = df[df[resolved_label_col].notnull()]

        if self.config.drop_duplicates:
            df = df.drop_duplicates(subset=[resolved_text_col, resolved_label_col])

        processed_df = df[[resolved_text_col, resolved_label_col]].rename(
            columns={
                resolved_text_col: self.config.default_text_column,
                resolved_label_col: self.config.default_label_column,
            }
        )

        return processed_df.reset_index(drop=True), (
            self.config.default_text_column,
            self.config.default_label_column,
        )

    def _resolve_text_column(self, df: pd.DataFrame, override: Optional[str]) -> str:
        if override and override in df.columns:
            return override
        if self.config.default_text_column in df.columns:
            return self.config.default_text_column

        # Fallback to the first object column
        for column in df.columns:
            if df[column].dtype == "object":
                return column

        raise ValueError("Unable to identify text column for dataset")

    def _resolve_label_column(self, df: pd.DataFrame, override: Optional[str]) -> str:
        if override and override in df.columns:
            return override
        if self.config.default_label_column in df.columns:
            return self.config.default_label_column

        # Look for intuitive column names
        for candidate in df.columns:
            if "label" in candidate.lower() or candidate.lower() in {"target", "class"}:
                return candidate

        # Create a new label column if none exists
        generated_name = self.config.default_label_column
        if generated_name not in df.columns:
            df[generated_name] = None
        return generated_name

    def _apply_auto_labeling(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str,
    ) -> pd.DataFrame:
        labels = df[label_column].copy()
        text_series = df[text_column].astype(str).str.lower()

        for rule in self.config.auto_labeling_rules:
            rule_keywords = [keyword.lower() for keyword in rule.keywords]
            mask = labels.isnull()
            if not mask.any():
                break
            if rule_keywords:
                pattern = "|".join(map(re.escape, rule_keywords))
                rule_mask = text_series.str.contains(pattern, case=False, na=False)
            else:
                rule_mask = pd.Series(False, index=df.index)
            labels = labels.where(~(mask & rule_mask), other=rule.label.lower())

        if labels.isnull().any() and self.config.default_label:
            labels = labels.fillna(self.config.default_label.lower())

        df[label_column] = labels
        return df

    def _validate(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        return self.validator.validate_dataset(
            dataframe,
            rules=self.config.quality_rules if self.config.quality_rules else None,
        )

    def _quality_passed(self, report: Dict[str, Any]) -> bool:
        score_ok = report.get("quality_score", 0.0) >= self.config.quality_score_threshold
        checks_ok = all(
            check.get("status") != "fail" for check in report.get("checks", {}).values()
        )
        return bool(score_ok and checks_ok)

    def _persist_outputs(
        self,
        *,
        dataset_id: str,
        dataset_name: str,
        dataframe: pd.DataFrame,
        quality_report: Dict[str, Any],
        text_column: str,
        label_column: str,
    ) -> Tuple[Path, Path, Path]:
        processed_dir = self.config.processed_path / dataset_id
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_file_path = processed_dir / "data.parquet"
        dataframe.to_parquet(processed_file_path, index=False)

        quality_report_path = self._write_quality_report(processed_dir, quality_report)

        registry_path = Path(
            self.registry.register_dataframe(
                dataframe,
                dataset_name=dataset_id,
                text_column=text_column,
                label_column=label_column,
                extra_metadata={
                    "quality_score": quality_report.get("quality_score"),
                    "quality_passed": True,
                    "quality_report_path": str(quality_report_path),
                },
            )
        )

        return processed_file_path, registry_path, quality_report_path

    def _execute_pipeline_from_hf_dataset(
        self,
        *,
        dataset_path: Path,
        dataset_id: str,
        dataset_name: str,
        text_column: Optional[str],
        label_column: Optional[str],
    ) -> DatasetPipelineResult:
        """Process a HuggingFace dataset directory directly.
        
        This method loads a HuggingFace dataset from disk and processes it through the
        quality pipeline, handling different dataset structures properly.
        """
        from datasets import load_from_disk, Dataset, DatasetDict

        # Ensure the dataset path exists
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
            
        # Load the dataset from disk with robust error handling
        try:
            print(f"Loading HuggingFace dataset from {dataset_path}...")
            hf_dataset = load_from_disk(str(dataset_path))
        except Exception as e:
            print(f"Error loading dataset from {dataset_path}: {e}")
            raise ValueError(f"Failed to load HuggingFace dataset from {dataset_path}: {e}")

        # Convert to pandas DataFrame with appropriate handling of dataset structure
        try:
            if isinstance(hf_dataset, DatasetDict):
                # DatasetDict - use train split if available, otherwise first split
                print(f"Dataset splits: {list(hf_dataset.keys())}")
                if 'train' in hf_dataset:
                    df = hf_dataset['train'].to_pandas()
                    print(f"Using 'train' split with {len(df)} rows")
                else:
                    first_key = list(hf_dataset.keys())[0]
                    df = hf_dataset[first_key].to_pandas()
                    print(f"Using '{first_key}' split with {len(df)} rows")
            elif isinstance(hf_dataset, Dataset):
                # Single Dataset
                df = hf_dataset.to_pandas()
                print(f"Using dataset with {len(df)} rows")
            else:
                raise TypeError(f"Unknown dataset type: {type(hf_dataset)}")
        except Exception as e:
            print(f"Error converting dataset to pandas DataFrame: {e}")
            raise ValueError(f"Failed to convert dataset to DataFrame: {e}")

        # Create uploads directory if it doesn't exist yet
        (self.config.uploads_path / dataset_id).mkdir(parents=True, exist_ok=True)

        # Process the dataframe through the regular pipeline
        try:
            processed_df, resolved_columns = self._transform_dataframe(
                df, text_column=text_column, label_column=label_column
            )
            print(f"Transformed dataframe: {len(processed_df)} rows, columns: {processed_df.columns}")
        except Exception as e:
            print(f"Error transforming dataframe: {e}")
            raise ValueError(f"Failed to transform dataset: {e}")

        # Validate quality
        try:
            quality_report = self._validate(processed_df)
            quality_passed = self._quality_passed(quality_report)
            print(f"Quality validation: passed={quality_passed}, score={quality_report.get('quality_score')}")
        except Exception as e:
            print(f"Error validating dataset quality: {e}")
            quality_report = {"quality_score": 0, "error": str(e), "issues": [{"type": "error", "message": str(e)}]}
            quality_passed = False

        processed_file_path: Optional[Path] = None
        registry_path: Optional[Path] = None

        # Persist results if quality passed
        try:
            if quality_passed:
                (
                    processed_file_path,
                    registry_path,
                    report_path,
                ) = self._persist_outputs(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    dataframe=processed_df,
                    quality_report=quality_report,
                    text_column=resolved_columns[0],
                    label_column=resolved_columns[1],
                )
                print(f"✓ Persisted processed dataset to {processed_file_path}")
            else:
                report_path = self._write_quality_report(
                    directory=self.config.uploads_path / dataset_id,
                    report=quality_report,
                )
                print(f"✓ Wrote quality report to {report_path}")
        except Exception as e:
            print(f"Error persisting processed dataset: {e}")
            report_path = self._write_quality_report(
                directory=self.config.uploads_path / dataset_id,
                report=quality_report,
            )

        return DatasetPipelineResult(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            raw_file_path=dataset_path,
            processed_file_path=processed_file_path,
            registry_path=registry_path,
            quality_report_path=report_path,
            rows=len(processed_df),
            columns=len(processed_df.columns),
            text_column=resolved_columns[0],
            label_column=resolved_columns[1],
            quality_report=quality_report,
            quality_passed=quality_passed,
        )

    def _write_quality_report(self, directory: Path, report: Dict[str, Any]) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / "quality_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report_path
