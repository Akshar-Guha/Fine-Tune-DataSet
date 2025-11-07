"""Base artifact schemas."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class ArtifactType(str, Enum):
    """Types of artifacts."""
    ADAPTER = "adapter"
    QUANTIZED = "quantized"
    INDEX = "index"
    DATASET = "dataset"
    SNN_MODEL = "snn_model"


class GovernanceStatus(str, Enum):
    """Artifact governance stages."""
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"
    ARCHIVED = "archived"


class AdapterConfig(BaseModel):
    """LoRA/QLoRA adapter configuration."""
    type: str = Field(..., description="qlora, lora, ia3, etc")
    rank: int = Field(..., description="Adapter rank")
    alpha: int = Field(..., description="Scaling parameter")
    target_modules: List[str] = Field(..., description="Modules to adapt")
    flash_attn: bool = Field(default=False, description="Use Flash Attention 2")
    dropout: float = Field(default=0.05, description="Dropout rate")


class BaseModelInfo(BaseModel):
    """Base model information."""
    name: str = Field(..., description="Model identifier")
    revision: str = Field(..., description="Model version/revision")
    sha256: str = Field(..., description="SHA256 checksum")
    size_bytes: int = Field(..., description="Model size in bytes")


class DatasetInfo(BaseModel):
    """Dataset information."""
    id: str = Field(..., description="Dataset identifier")
    delta_version: str = Field(..., description="Delta Lake version")
    delta_snapshot_id: int = Field(..., description="Snapshot ID")
    sha256: str = Field(..., description="Dataset checksum")
    num_rows: int = Field(..., description="Number of rows")
    num_columns: int = Field(..., description="Number of columns")


class TrainingInfo(BaseModel):
    """Training run information."""
    workflow_id: str = Field(..., description="Temporal workflow ID")
    temporal_run_id: str = Field(..., description="Temporal run ID")
    mlflow_run_id: str = Field(..., description="MLflow run ID")
    duration_seconds: int = Field(..., description="Training duration")
    gpu_type: str = Field(..., description="GPU type used")
    num_gpus: int = Field(..., description="Number of GPUs")
    deepspeed_config: Optional[Dict] = Field(None, description="DeepSpeed config")
    total_steps: int = Field(..., description="Total training steps")
    final_loss: float = Field(..., description="Final training loss")


class QuantizationInfo(BaseModel):
    """Quantization information."""
    method: str = Field(..., description="awq, gptq, hqq, gguf")
    bits: int = Field(..., description="Quantization bits")
    group_size: Optional[int] = Field(None, description="Quantization group size")
    calibration_dataset: Optional[str] = Field(None, description="Calibration data")
    perplexity_delta: float = Field(..., description="Perplexity change from FP16")


class ProvenanceInfo(BaseModel):
    """Artifact provenance tracking."""
    parent_artifact_id: Optional[str] = Field(None, description="Parent artifact")
    algorithm_plugins: List[str] = Field(
        default_factory=list, description="Plugin IDs used"
    )
    git_commit: str = Field(..., description="Git commit hash")
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(..., description="Creator user ID")
    trace_id: str = Field(..., description="OpenTelemetry trace ID")


class GovernanceInfo(BaseModel):
    """Artifact governance information."""
    status: GovernanceStatus = Field(default=GovernanceStatus.DEV)
    promoted_by: Optional[str] = Field(None, description="Promoter user ID")
    promoted_at: Optional[datetime] = Field(None, description="Promotion timestamp")
    promotion_checklist: Dict[str, bool] = Field(
        default_factory=dict, description="Checklist items"
    )
    approvers: List[str] = Field(default_factory=list, description="Approver list")


class SignatureInfo(BaseModel):
    """Artifact cryptographic signature."""
    algorithm: str = Field(default="ed25519", description="Signature algorithm")
    public_key: str = Field(..., description="Base64-encoded public key")
    signature: str = Field(..., description="Base64-encoded signature")
    signed_at: datetime = Field(default_factory=datetime.now)


class ArtifactManifest(BaseModel):
    """Complete artifact manifest with lineage and governance."""
    artifact_id: str = Field(..., description="Unique artifact ID")
    name: str = Field(..., description="Human-readable name")
    framework: str = Field(default="pytorch", description="ML framework")
    type: ArtifactType = Field(..., description="Artifact type")
    
    # Type-specific configs
    adapter: Optional[AdapterConfig] = None
    quantization: Optional[QuantizationInfo] = None
    
    # Related artifacts
    base_model: BaseModelInfo
    dataset: DatasetInfo
    
    # Training metadata
    training: TrainingInfo
    
    # Metrics
    metrics: Dict[str, float] = Field(default_factory=dict)
    
    # Storage
    storage_uri: str = Field(..., description="Storage location")
    delta_table_uri: Optional[str] = Field(None, description="Delta Lake URI")
    size_bytes: int = Field(..., description="Artifact size")
    
    # Provenance
    provenance: ProvenanceInfo
    
    # Governance
    governance: GovernanceInfo
    
    # Security
    signature: SignatureInfo
    
    # Tags for search
    tags: List[str] = Field(default_factory=list)
    
    # Documentation
    description: Optional[str] = Field(None, description="Description")
    readme: Optional[str] = Field(None, description="README content")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
