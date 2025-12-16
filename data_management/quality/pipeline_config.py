"""Dataset pipeline configuration and helpers."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class AutoLabelingRule:
    """Keyword-based heuristic rule for auto labeling."""

    label: str
    keywords: List[str]


@dataclass
class DatasetPipelineConfig:
    """Configuration for dataset ingestion pipeline."""

    dataset_dir: str = "./datasets"
    uploads_subdir: str = "uploads"
    processed_subdir: str = "processed"
    default_text_column: str = "text"
    default_label_column: str = "label"
    registry_subdir: str = "registry"
    required_columns: List[str] = field(default_factory=lambda: ["text", "label"])
    quality_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    quality_score_threshold: float = 70.0
    drop_missing_text: bool = True
    drop_missing_label: bool = True
    strip_text: bool = True
    lowercase_labels: bool = True
    allowed_label_values: Optional[List[str]] = None
    auto_labeling_rules: List[AutoLabelingRule] = field(default_factory=list)
    default_label: str = "unspecified"
    enable_auto_labeling: bool = False
    drop_duplicates: bool = True

    @property
    def uploads_path(self) -> Path:
        return Path(self.dataset_dir) / self.uploads_subdir

    @property
    def processed_path(self) -> Path:
        return Path(self.dataset_dir) / self.processed_subdir

    @property
    def registry_path(self) -> Path:
        return Path(self.dataset_dir) / self.registry_subdir


DEFAULT_CONFIG = DatasetPipelineConfig(
    quality_rules={
        "text": {
            "min_length": 5,
        }
    },
    quality_score_threshold=80.0,
    auto_labeling_rules=[
        AutoLabelingRule(label="positive", keywords=["good", "great", "excellent", "love", "amazing"]),
        AutoLabelingRule(label="negative", keywords=["bad", "terrible", "awful", "hate", "poor"]),
    ],
    enable_auto_labeling=True,
    allowed_label_values=["positive", "negative", "unspecified"],
)


def _load_external_config() -> Optional[DatasetPipelineConfig]:
    """Load pipeline config from JSON file when path is provided."""

    config_path = os.getenv("DATASET_PIPELINE_CONFIG")
    if not config_path:
        return None

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset pipeline config not found at {config_path}")

    with path.open("r", encoding="utf-8") as config_file:
        raw = json.load(config_file)

    auto_rules = [
        AutoLabelingRule(label=rule["label"], keywords=rule.get("keywords", []))
        for rule in raw.get("auto_labeling_rules", [])
    ]

    return DatasetPipelineConfig(
        dataset_dir=raw.get("dataset_dir", DEFAULT_CONFIG.dataset_dir),
        uploads_subdir=raw.get("uploads_subdir", DEFAULT_CONFIG.uploads_subdir),
        processed_subdir=raw.get("processed_subdir", DEFAULT_CONFIG.processed_subdir),
        registry_subdir=raw.get("registry_subdir", DEFAULT_CONFIG.registry_subdir),
        default_text_column=raw.get("default_text_column", DEFAULT_CONFIG.default_text_column),
        default_label_column=raw.get("default_label_column", DEFAULT_CONFIG.default_label_column),
        required_columns=raw.get("required_columns", DEFAULT_CONFIG.required_columns),
        quality_rules=raw.get("quality_rules", DEFAULT_CONFIG.quality_rules),
        quality_score_threshold=raw.get("quality_score_threshold", DEFAULT_CONFIG.quality_score_threshold),
        drop_missing_text=raw.get("drop_missing_text", DEFAULT_CONFIG.drop_missing_text),
        drop_missing_label=raw.get("drop_missing_label", DEFAULT_CONFIG.drop_missing_label),
        strip_text=raw.get("strip_text", DEFAULT_CONFIG.strip_text),
        lowercase_labels=raw.get("lowercase_labels", DEFAULT_CONFIG.lowercase_labels),
        allowed_label_values=raw.get("allowed_label_values", DEFAULT_CONFIG.allowed_label_values),
        auto_labeling_rules=auto_rules or DEFAULT_CONFIG.auto_labeling_rules,
        default_label=raw.get("default_label", DEFAULT_CONFIG.default_label),
        enable_auto_labeling=raw.get("enable_auto_labeling", DEFAULT_CONFIG.enable_auto_labeling),
    )


def load_pipeline_config() -> DatasetPipelineConfig:
    """Return dataset pipeline config, preferring external definition."""

    external = _load_external_config()
    config = external or DEFAULT_CONFIG
    
    # Create base dataset directory first
    try:
        Path(config.dataset_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to create dataset directory {config.dataset_dir}: {e}")
        
    # Then create all subdirectories
    for path_property in ["uploads_path", "processed_path", "registry_path"]:
        try:
            directory = getattr(config, path_property)
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created dataset directory: {directory}")
        except Exception as e:
            print(f"Warning: Failed to create {path_property}: {e}")
    
    return config
