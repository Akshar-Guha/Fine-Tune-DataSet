"""Pydantic models for API request/response validation."""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class LLMVersionStatus(str, Enum):
    """Status of a model version."""
    DRAFT = "draft"
    TRAINING = "training"
    COMPLETED = "completed"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class DatasetVersionStatus(str, Enum):
    """Status of a dataset version."""
    DRAFT = "draft"
    PROCESSING = "processing"
    READY = "ready"
    ARCHIVED = "archived"
    FAILED = "failed"


# Base Models
class UserBase(BaseModel):
    """Base user model."""
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_admin: bool = False


class UserCreate(UserBase):
    """User create model."""
    password: str


class UserUpdate(BaseModel):
    """User update model."""
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    password: Optional[str] = None


class User(UserBase):
    """User response model."""
    id: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        orm_mode = True


class LLMBase(BaseModel):
    """Base LLM model."""
    name: str
    description: Optional[str] = None
    base_model: str


class LLMCreate(LLMBase):
    """LLM create model."""
    pass


class LLMUpdate(BaseModel):
    """LLM update model."""
    name: Optional[str] = None
    description: Optional[str] = None
    base_model: Optional[str] = None


class LLM(LLMBase):
    """LLM response model."""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class LLMWithVersions(LLM):
    """LLM with versions response model."""
    versions: List["LLMVersion"] = []

    class Config:
        orm_mode = True


class TrainingParameterBase(BaseModel):
    """Base training parameter model."""
    name: str
    description: Optional[str] = None
    type: str
    default_value: Optional[str] = None
    min_value: Optional[str] = None
    max_value: Optional[str] = None


class TrainingParameterCreate(TrainingParameterBase):
    """Training parameter create model."""
    pass


class TrainingParameter(TrainingParameterBase):
    """Training parameter response model."""
    id: str
    created_at: datetime

    class Config:
        orm_mode = True


class LLMVersionBase(BaseModel):
    """Base LLM version model."""
    version: str
    status: LLMVersionStatus = LLMVersionStatus.DRAFT
    training_dataset_version_id: Optional[str] = None
    validation_dataset_version_id: Optional[str] = None
    model_size: Optional[int] = None
    checkpoint_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    training_metrics: Optional[Dict[str, Any]] = None
    validation_metrics: Optional[Dict[str, Any]] = None


class LLMVersionCreate(LLMVersionBase):
    """LLM version create model."""
    llm_id: str
    created_by: str
    parameter_ids: Optional[List[str]] = None
    tag_names: Optional[List[str]] = None


class LLMVersionUpdate(BaseModel):
    """LLM version update model."""
    status: Optional[LLMVersionStatus] = None
    training_dataset_version_id: Optional[str] = None
    validation_dataset_version_id: Optional[str] = None
    model_size: Optional[int] = None
    checkpoint_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    training_metrics: Optional[Dict[str, Any]] = None
    validation_metrics: Optional[Dict[str, Any]] = None
    parameter_ids: Optional[List[str]] = None
    tag_names: Optional[List[str]] = None


class LLMVersion(LLMVersionBase):
    """LLM version response model."""
    id: str
    llm_id: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    parameters: List[TrainingParameter] = []
    tags: List["Tag"] = []

    class Config:
        orm_mode = True


class DatasetBase(BaseModel):
    """Base dataset model."""
    name: str
    description: Optional[str] = None
    source: Optional[str] = None


class DatasetCreate(DatasetBase):
    """Dataset create model."""
    pass


class DatasetUpdate(BaseModel):
    """Dataset update model."""
    name: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None


class Dataset(DatasetBase):
    """Dataset response model."""
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class DatasetWithVersions(Dataset):
    """Dataset with versions response model."""
    versions: List["DatasetVersion"] = []

    class Config:
        orm_mode = True


class DatasetVersionBase(BaseModel):
    """Base dataset version model."""
    version: str
    status: DatasetVersionStatus = DatasetVersionStatus.DRAFT
    file_path: Optional[str] = None
    file_format: Optional[str] = None
    num_records: Optional[int] = None
    size_bytes: Optional[int] = None
    schema: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    statistics: Optional[Dict[str, Any]] = None


class DatasetVersionCreate(DatasetVersionBase):
    """Dataset version create model."""
    dataset_id: str
    created_by: str
    tag_names: Optional[List[str]] = None


class DatasetVersionUpdate(BaseModel):
    """Dataset version update model."""
    status: Optional[DatasetVersionStatus] = None
    file_path: Optional[str] = None
    file_format: Optional[str] = None
    num_records: Optional[int] = None
    size_bytes: Optional[int] = None
    schema: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    statistics: Optional[Dict[str, Any]] = None
    tag_names: Optional[List[str]] = None


class DatasetVersion(DatasetVersionBase):
    """Dataset version response model."""
    id: str
    dataset_id: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    tags: List["Tag"] = []

    class Config:
        orm_mode = True


class TrainingLogBase(BaseModel):
    """Base training log model."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    epoch: Optional[int] = None
    step: Optional[int] = None
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    additional_metrics: Optional[Dict[str, Any]] = None


class TrainingLogCreate(TrainingLogBase):
    """Training log create model."""
    llm_version_id: str


class TrainingLogCreateBatch(BaseModel):
    """Batch training log create model."""
    logs: List[TrainingLogCreate]


class TrainingLog(TrainingLogBase):
    """Training log response model."""
    id: str
    llm_version_id: str

    class Config:
        orm_mode = True


class InferenceLogBase(BaseModel):
    """Base inference log model."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    latency_ms: float
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    additional_data: Optional[Dict[str, Any]] = None


class InferenceLogCreate(InferenceLogBase):
    """Inference log create model."""
    llm_version_id: str


class InferenceLog(InferenceLogBase):
    """Inference log response model."""
    id: str
    llm_version_id: str

    class Config:
        orm_mode = True


class TagBase(BaseModel):
    """Base tag model."""
    name: str
    description: Optional[str] = None


class TagCreate(TagBase):
    """Tag create model."""
    pass


class Tag(TagBase):
    """Tag response model."""
    id: str
    created_at: datetime

    class Config:
        orm_mode = True


class AuditLogBase(BaseModel):
    """Base audit log model."""
    user_id: Optional[str] = None
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class AuditLogCreate(AuditLogBase):
    """Audit log create model."""
    pass


class AuditLog(AuditLogBase):
    """Audit log response model."""
    id: str
    timestamp: datetime

    class Config:
        orm_mode = True


class AlertBase(BaseModel):
    """Base alert model."""
    severity: str
    title: str
    message: str
    source: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    status: str = "open"
    details: Optional[Dict[str, Any]] = None


class AlertCreate(AlertBase):
    """Alert create model."""
    pass


class AlertUpdate(BaseModel):
    """Alert update model."""
    status: Optional[str] = None
    resolved_by: Optional[str] = None


class Alert(AlertBase):
    """Alert response model."""
    id: str
    timestamp: datetime
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    class Config:
        orm_mode = True


class SystemMetricBase(BaseModel):
    """Base system metric model."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metric_name: str
    metric_value: float
    metric_unit: Optional[str] = None
    hostname: Optional[str] = None
    service: Optional[str] = None
    additional_dimensions: Optional[Dict[str, Any]] = None


class SystemMetricCreate(SystemMetricBase):
    """System metric create model."""
    pass


class SystemMetricCreateBatch(BaseModel):
    """Batch system metric create model."""
    metrics: List[SystemMetricCreate]


class SystemMetric(SystemMetricBase):
    """System metric response model."""
    id: str

    class Config:
        orm_mode = True


# API key models
class APIKeyBase(BaseModel):
    """Base API key model."""
    name: str
    user_id: str
    expires_at: Optional[datetime] = None
    permissions: Optional[List[str]] = None


class APIKeyCreate(APIKeyBase):
    """API key create model."""
    pass


class APIKeyResponse(BaseModel):
    """API key response after creation."""
    id: str
    name: str
    prefix: str
    key: str  # Full key, only returned once at creation
    user_id: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    permissions: Optional[List[str]] = None


class APIKey(BaseModel):
    """API key response model (without the full key)."""
    id: str
    name: str
    prefix: str
    user_id: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool
    permissions: Optional[List[str]] = None

    class Config:
        orm_mode = True


# Token models for authentication
class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    """Token payload model."""
    sub: Optional[str] = None
    exp: Optional[int] = None


# Metrics and Analytics Models
class ModelMetrics(BaseModel):
    """Model metrics response model."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    history: List[Dict[str, Any]] = []
    confusion_matrix: Dict[str, Any]


# Error responses
class HTTPError(BaseModel):
    """HTTP error response model."""
    detail: str


# Forward references for models with circular dependencies
LLMWithVersions.update_forward_refs()
