"""Database models for storing ModelOps data and metrics."""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, Table, Text, JSON, Enum as SQLEnum, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


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


# Association tables for many-to-many relationships
llm_version_parameter_association = Table(
    'llm_version_parameter_association', Base.metadata,
    Column('llm_version_id', String(36), ForeignKey('llm_versions.id')),
    Column('parameter_id', String(36), ForeignKey('training_parameters.id'))
)

dataset_version_tag_association = Table(
    'dataset_version_tag_association', Base.metadata,
    Column('dataset_version_id', String(36), ForeignKey('dataset_versions.id')),
    Column('tag_id', String(36), ForeignKey('tags.id'))
)

llm_version_tag_association = Table(
    'llm_version_tag_association', Base.metadata,
    Column('llm_version_id', String(36), ForeignKey('llm_versions.id')),
    Column('tag_id', String(36), ForeignKey('tags.id'))
)


class User(Base):
    """User model for authentication and tracking."""
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    llm_versions = relationship("LLMVersion", back_populates="created_by_user")
    dataset_versions = relationship("DatasetVersion", back_populates="created_by_user")
    audit_logs = relationship("AuditLog", back_populates="user")


class LLM(Base):
    """Base LLM model definition."""
    __tablename__ = 'llms'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    base_model = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    versions = relationship("LLMVersion", back_populates="llm", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<LLM(name='{self.name}', base_model='{self.base_model}')>"


class LLMVersion(Base):
    """Specific version of an LLM with metrics and lineage."""
    __tablename__ = 'llm_versions'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    llm_id = Column(String(36), ForeignKey('llms.id'), nullable=False)
    version = Column(String(20), nullable=False)
    status = Column(SQLEnum(LLMVersionStatus), default=LLMVersionStatus.DRAFT)
    created_by = Column(String(36), ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Training metadata
    training_dataset_version_id = Column(String(36), ForeignKey('dataset_versions.id'))
    validation_dataset_version_id = Column(String(36), ForeignKey('dataset_versions.id'))
    model_size = Column(BigInteger)  # Size in bytes
    checkpoint_path = Column(String(500))
    config = Column(JSON)  # JSON blob of configuration
    
    # Metrics - stored as JSON for flexibility
    training_metrics = Column(JSON)  # loss, perplexity, etc.
    validation_metrics = Column(JSON)  # evaluation results
    
    # Relationships
    llm = relationship("LLM", back_populates="versions")
    created_by_user = relationship("User", back_populates="llm_versions")
    parameters = relationship("TrainingParameter", secondary=llm_version_parameter_association)
    training_logs = relationship("TrainingLog", back_populates="llm_version", cascade="all, delete-orphan")
    inference_logs = relationship("InferenceLog", back_populates="llm_version", cascade="all, delete-orphan")
    training_dataset_version = relationship("DatasetVersion", foreign_keys=[training_dataset_version_id])
    validation_dataset_version = relationship("DatasetVersion", foreign_keys=[validation_dataset_version_id])
    tags = relationship("Tag", secondary=llm_version_tag_association)
    
    __table_args__ = (
        # Ensure (llm_id, version) is unique
        {'unique_constraint_name': 'unique_llm_version'},
    )
    
    def __repr__(self):
        return f"<LLMVersion(llm='{self.llm.name}', version='{self.version}', status='{self.status}')>"


class Dataset(Base):
    """Dataset definition that can have multiple versions."""
    __tablename__ = 'datasets'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    source = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Dataset(name='{self.name}', source='{self.source}')>"


class DatasetVersion(Base):
    """Specific version of a dataset with metrics."""
    __tablename__ = 'dataset_versions'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String(36), ForeignKey('datasets.id'), nullable=False)
    version = Column(String(20), nullable=False)
    status = Column(SQLEnum(DatasetVersionStatus), default=DatasetVersionStatus.DRAFT)
    created_by = Column(String(36), ForeignKey('users.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Dataset metadata
    file_path = Column(String(500))
    file_format = Column(String(50))
    num_records = Column(Integer)
    size_bytes = Column(BigInteger)
    schema = Column(JSON)  # JSON representation of schema
    quality_score = Column(Float)  # 0-100 quality score
    statistics = Column(JSON)  # Statistical profile of data
    
    # Relationships
    dataset = relationship("Dataset", back_populates="versions")
    created_by_user = relationship("User", back_populates="dataset_versions")
    tags = relationship("Tag", secondary=dataset_version_tag_association)
    
    __table_args__ = (
        # Ensure (dataset_id, version) is unique
        {'unique_constraint_name': 'unique_dataset_version'},
    )
    
    def __repr__(self):
        return f"<DatasetVersion(dataset='{self.dataset.name}', version='{self.version}', status='{self.status}')>"


class TrainingParameter(Base):
    """Training hyperparameters that can be reused across model versions."""
    __tablename__ = 'training_parameters'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    type = Column(String(50), nullable=False)  # e.g. "float", "int", "str", "bool"
    default_value = Column(String(255))
    min_value = Column(String(255))
    max_value = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<TrainingParameter(name='{self.name}', type='{self.type}')>"


class TrainingLog(Base):
    """Detailed logs from training runs."""
    __tablename__ = 'training_logs'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    llm_version_id = Column(String(36), ForeignKey('llm_versions.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    epoch = Column(Integer)
    step = Column(Integer)
    loss = Column(Float)
    learning_rate = Column(Float)
    additional_metrics = Column(JSON)  # Other metrics as JSON
    
    # Relationships
    llm_version = relationship("LLMVersion", back_populates="training_logs")
    
    def __repr__(self):
        return f"<TrainingLog(epoch={self.epoch}, step={self.step}, loss={self.loss})>"


class InferenceLog(Base):
    """Logs of model inference with inputs, outputs and metrics."""
    __tablename__ = 'inference_logs'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    llm_version_id = Column(String(36), ForeignKey('llm_versions.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    input_text = Column(Text)
    output_text = Column(Text)
    latency_ms = Column(Float)
    tokens_input = Column(Integer)
    tokens_output = Column(Integer)
    additional_data = Column(JSON)  # Additional context or metadata
    
    # Relationships
    llm_version = relationship("LLMVersion", back_populates="inference_logs")
    
    def __repr__(self):
        return f"<InferenceLog(llm_version_id='{self.llm_version_id}', latency_ms={self.latency_ms})>"


class Tag(Base):
    """Tags for datasets and models."""
    __tablename__ = 'tags'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Tag(name='{self.name}')>"


class JobType(str, Enum):
    """Types of jobs the platform can run."""
    SFT_TRAINING = "sft_training"
    QUANTIZATION = "quantization"
    RAG_INDEXING = "rag_indexing"
    RLHF = "rlhf"
    EVALUATION = "evaluation"


class JobStatus(str, Enum):
    """Lifecycle states for jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    """Track submitted jobs and their execution metadata."""
    __tablename__ = 'jobs'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    job_type = Column(SQLEnum(JobType), nullable=False)
    status = Column(SQLEnum(JobStatus), nullable=False, default=JobStatus.PENDING)
    config = Column(JSON, nullable=False)
    dataset_id = Column(String(255), nullable=False)
    base_model = Column(String(255))
    priority = Column(Integer, default=0)
    workflow_id = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error = Column(Text)
    metrics = Column(JSON)
    artifacts = Column(JSON)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return (
            f"<Job(name='{self.name}', type='{self.job_type}', status='{self.status}', "
            f"dataset_id='{self.dataset_id}')>"
        )


class AuditLog(Base):
    """Audit logs for security and compliance."""
    __tablename__ = 'audit_logs'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(36))
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    details = Column(JSON)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    def __repr__(self):
        return f"<AuditLog(action='{self.action}', resource_type='{self.resource_type}', timestamp='{self.timestamp}')>"


class SystemMetric(Base):
    """System-level metrics for monitoring."""
    __tablename__ = 'system_metrics'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))
    hostname = Column(String(255))
    service = Column(String(100))
    additional_dimensions = Column(JSON)
    
    def __repr__(self):
        return f"<SystemMetric(metric_name='{self.metric_name}', metric_value={self.metric_value})>"


class Alert(Base):
    """System and model alerts."""
    __tablename__ = 'alerts'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    source = Column(String(100), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(String(36))
    status = Column(String(20), default="open")  # open, acknowledged, resolved
    resolved_at = Column(DateTime)
    resolved_by = Column(String(36), ForeignKey('users.id'))
    details = Column(JSON)
    
    def __repr__(self):
        return f"<Alert(severity='{self.severity}', title='{self.title}', status='{self.status}')>"


class APIKey(Base):
    """API keys for authentication."""
    __tablename__ = 'api_keys'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    name = Column(String(255), nullable=False)
    prefix = Column(String(10), nullable=False)
    hashed_key = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    last_used_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    permissions = Column(JSON)  # JSON array of permissions
    
    def __repr__(self):
        return f"<APIKey(name='{self.name}', prefix='{self.prefix}')>"
