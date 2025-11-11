"""
Repository pattern implementation for database operations.
Provides CRUD operations for all database models.
"""

from typing import List, Dict, Any, Optional, Generic, TypeVar, Type
from uuid import uuid4
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy import desc
from datetime import datetime

from .models import (
    User,
    LLM,
    LLMVersion,
    Dataset,
    DatasetVersion,
    Job,
    JobStatus,
    TrainingParameter,
    TrainingLog,
    InferenceLog,
    Tag,
    AuditLog,
    SystemMetric,
    Alert,
    APIKey,
)

# Define generic type for models
T = TypeVar('T')


class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations."""

    def __init__(self, model: Type[T], db: Session):
        self.model = model
        self.db = db
    
    def get_by_id(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        return self.db.query(self.model).filter(self.model.id == id).first()
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """Get all entities with pagination."""
        return self.db.query(self.model).offset(skip).limit(limit).all()
    
    def create(self, obj_in: Dict[str, Any]) -> T:
        """Create entity."""
        obj_id = obj_in.get('id') or str(uuid4())
        obj_in['id'] = obj_id
        db_obj = self.model(**obj_in)
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj
    
    def update(self, id: str, obj_in: Dict[str, Any]) -> Optional[T]:
        """Update entity."""
        db_obj = self.get_by_id(id)
        if not db_obj:
            return None
            
        for key, value in obj_in.items():
            if hasattr(db_obj, key):
                setattr(db_obj, key, value)
        
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj
    
    def delete(self, id: str) -> bool:
        """Delete entity."""
        db_obj = self.get_by_id(id)
        if not db_obj:
            return False
            
        self.db.delete(db_obj)
        self.db.commit()
        return True


class UserRepository(BaseRepository[User]):
    """Repository for User model."""
    
    def __init__(self, db: Session):
        super().__init__(User, db)
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.db.query(self.model).filter(self.model.username == username).first()
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(self.model).filter(self.model.email == email).first()
    
    def update_last_login(self, user_id: str) -> None:
        """Update user's last login time."""
        user = self.get_by_id(user_id)
        if user:
            user.last_login = datetime.utcnow()
            self.db.commit()


class LLMRepository(BaseRepository[LLM]):
    """Repository for LLM model."""
    
    def __init__(self, db: Session):
        super().__init__(LLM, db)
    
    def get_by_name(self, name: str) -> Optional[LLM]:
        """Get LLM by name."""
        return self.db.query(self.model).filter(self.model.name == name).first()
    
    def get_with_versions(self, id: str) -> Optional[LLM]:
        """Get LLM with all its versions."""
        return self.db.query(self.model).filter(self.model.id == id).first()


class LLMVersionRepository(BaseRepository[LLMVersion]):
    """Repository for LLMVersion model."""
    
    def __init__(self, db: Session):
        super().__init__(LLMVersion, db)
    
    def get_by_llm_and_version(self, llm_id: str, version: str) -> Optional[LLMVersion]:
        """Get LLMVersion by LLM ID and version."""
        return (
            self.db.query(self.model)
            .filter(
                self.model.llm_id == llm_id,
                self.model.version == version
            )
            .first()
        )
    
    def get_latest_version(self, llm_id: str) -> Optional[LLMVersion]:
        """Get latest version of a specific LLM."""
        return (
            self.db.query(self.model)
            .filter(self.model.llm_id == llm_id)
            .order_by(desc(self.model.created_at))
            .first()
        )
    
    def get_versions_by_llm(self, llm_id: str, skip: int = 0, limit: int = 100) -> List[LLMVersion]:
        """Get all versions of a specific LLM."""
        return (
            self.db.query(self.model)
            .filter(self.model.llm_id == llm_id)
            .order_by(desc(self.model.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )


class DatasetRepository(BaseRepository[Dataset]):
    """Repository for Dataset model."""
    
    def __init__(self, db: Session):
        super().__init__(Dataset, db)
    
    def get_by_name(self, name: str) -> Optional[Dataset]:
        """Get dataset by name."""
        return self.db.query(self.model).filter(self.model.name == name).first()


class DatasetVersionRepository(BaseRepository[DatasetVersion]):
    """Repository for DatasetVersion model."""
    
    def __init__(self, db: Session):
        super().__init__(DatasetVersion, db)
    
    def get_by_dataset_and_version(self, dataset_id: str, version: str) -> Optional[DatasetVersion]:
        """Get DatasetVersion by Dataset ID and version."""
        return (
            self.db.query(self.model)
            .filter(
                self.model.dataset_id == dataset_id,
                self.model.version == version
            )
            .first()
        )
    
    def get_latest_version(self, dataset_id: str) -> Optional[DatasetVersion]:
        """Get latest version of a specific Dataset."""
        return (
            self.db.query(self.model)
            .filter(self.model.dataset_id == dataset_id)
            .order_by(desc(self.model.created_at))
            .first()
        )
    
    def get_versions_by_dataset(self, dataset_id: str, skip: int = 0, limit: int = 100) -> List[DatasetVersion]:
        """Get all versions of a specific Dataset."""
        return (
            self.db.query(self.model)
            .filter(self.model.dataset_id == dataset_id)
            .order_by(desc(self.model.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )


class JobRepository(BaseRepository[Job]):
    """Repository for Job model with Temporal integration."""

    def __init__(self, db: Session):
        super().__init__(Job, db)
        # Lazy import to avoid circular imports
        from workflows.temporal_client import temporal_client
        self.temporal_client = temporal_client
        
    async def create_job(self, obj_in: Dict[str, Any]) -> Job:
        """Create a job with provided payload and submit to Temporal."""
        # First persist to database
        job = self.create(obj_in)
        
        try:
            # Submit to appropriate workflow based on job type
            if job.job_type == JobType.SFT_TRAINING:
                # Convert config to proper QLoRAConfig
                from workflows.sft.config import QLoRAConfig
                config = QLoRAConfig(**job.config)
                workflow_id = await self.temporal_client.submit_qlora_job(
                    job.id, config
                )
            elif job.job_type == JobType.QUANTIZATION:
                workflow_id = await self.temporal_client.submit_quantization_job(
                    job.id, job.config
                )
            elif job.job_type == JobType.RAG_INDEXING:
                workflow_id = await self.temporal_client.submit_rag_indexing_job(
                    job.id, job.config
                )
            else:
                # Other job types not yet implemented
                # Just return the job without submitting to Temporal
                return job
                
            # Update job with workflow ID
            self.update(job.id, {"workflow_id": workflow_id})
            return job
            
        except Exception as e:
            # Update job status to failed if workflow submission fails
            self.update(job.id, {
                "status": JobStatus.FAILED,
                "error": f"Failed to submit workflow: {str(e)}"
            })
            raise

    def list_jobs(
        self,
        *,
        job_type: Optional[str] = None,
        status: Optional[JobStatus] = None,
        skip: int = 0,
        limit: int = 100
    ) -> tuple[list[Job], int]:
        """List jobs with optional filters."""
        query = self.db.query(self.model)
        if job_type:
            query = query.filter(self.model.job_type == job_type)
        if status:
            query = query.filter(self.model.status == status)

        total = query.count()
        jobs = (
            query.order_by(desc(self.model.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
        return jobs, total
        
    def get_by_id(self, job_id: str) -> Optional[Job]:
        """Get job by ID, possibly updating status from Temporal."""
        job = super().get_by_id(job_id)
        
        if job and job.workflow_id and job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            # Try to update status from Temporal
            self._update_job_status_from_temporal(job)
            
        return job

    async def _update_job_status_from_temporal(self, job: Job) -> None:
        """Update job status from Temporal."""
        try:
            if not job.workflow_id:
                return
                
            workflow_status = await self.temporal_client.get_workflow_status(job.workflow_id)
            
            status_mapping = {
                "RUNNING": JobStatus.RUNNING,
                "COMPLETED": JobStatus.COMPLETED,
                "FAILED": JobStatus.FAILED,
                "CANCELED": JobStatus.CANCELLED,
                "TIMED_OUT": JobStatus.FAILED
            }
            
            if workflow_status["status"] in status_mapping:
                new_status = status_mapping[workflow_status["status"]]
                
                # Only update if changed
                if job.status != new_status:
                    updates = {"status": new_status}
                    
                    # Set timestamps
                    if new_status == JobStatus.RUNNING and not job.started_at:
                        updates["started_at"] = datetime.utcnow()
                    elif new_status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and not job.completed_at:
                        updates["completed_at"] = datetime.utcnow()
                        
                    # Update metrics if workflow completed
                    if new_status == JobStatus.COMPLETED:
                        # TODO: Fetch result from Temporal
                        pass
                    
                    self.update(job.id, updates)
        except Exception as e:
            # Log error but don't fail the request
            import logging
            logging.error(f"Failed to update job status from Temporal: {e}")

    def update_job(self, job_id: str, obj_in: Dict[str, Any]) -> Optional[Job]:
        """Update job fields."""
        return self.update(job_id, obj_in)

    async def cancel_job(self, job_id: str) -> Optional[Job]:
        """Cancel job in Temporal and mark as cancelled."""
        job = self.get_by_id(job_id)
        
        if job and job.workflow_id:
            try:
                # Cancel in Temporal
                await self.temporal_client.cancel_workflow(job.workflow_id)
            except Exception as e:
                import logging
                logging.error(f"Failed to cancel workflow {job.workflow_id}: {e}")
                
        # Update database
        payload: Dict[str, Any] = {
            "status": JobStatus.CANCELLED,
            "completed_at": datetime.utcnow(),
        }
        return self.update(job_id, payload)


class TrainingLogRepository(BaseRepository[TrainingLog]):
    """Repository for TrainingLog model."""
    
    def __init__(self, db: Session):
        super().__init__(TrainingLog, db)
    
    def get_by_llm_version(self, llm_version_id: str, skip: int = 0, limit: int = 100) -> List[TrainingLog]:
        """Get training logs for a specific LLM version."""
        return (
            self.db.query(self.model)
            .filter(self.model.llm_version_id == llm_version_id)
            .order_by(self.model.timestamp)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def create_batch(self, logs: List[Dict[str, Any]]) -> List[TrainingLog]:
        """Batch create training logs."""
        db_logs = [self.model(**log) for log in logs]
        self.db.add_all(db_logs)
        self.db.commit()
        return db_logs


class InferenceLogRepository(BaseRepository[InferenceLog]):
    """Repository for InferenceLog model."""
    
    def __init__(self, db: Session):
        super().__init__(InferenceLog, db)
    
    def get_by_llm_version(self, llm_version_id: str, skip: int = 0, limit: int = 100) -> List[InferenceLog]:
        """Get inference logs for a specific LLM version."""
        return (
            self.db.query(self.model)
            .filter(self.model.llm_version_id == llm_version_id)
            .order_by(desc(self.model.timestamp))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_average_latency(self, llm_version_id: str) -> float:
        """Get average latency for a specific LLM version."""
        result = (
            self.db.query(self.model)
            .filter(self.model.llm_version_id == llm_version_id)
            .with_entities(self.model.latency_ms)
            .all()
        )
        latencies = [r.latency_ms for r in result]
        return sum(latencies) / len(latencies) if latencies else 0


class AuditLogRepository(BaseRepository[AuditLog]):
    """Repository for AuditLog model."""
    
    def __init__(self, db: Session):
        super().__init__(AuditLog, db)
    
    def get_by_user(self, user_id: str, skip: int = 0, limit: int = 100) -> List[AuditLog]:
        """Get audit logs for a specific user."""
        return (
            self.db.query(self.model)
            .filter(self.model.user_id == user_id)
            .order_by(desc(self.model.timestamp))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_resource(self, resource_type: str, resource_id: str, skip: int = 0, limit: int = 100) -> List[AuditLog]:
        """Get audit logs for a specific resource."""
        return (
            self.db.query(self.model)
            .filter(
                self.model.resource_type == resource_type,
                self.model.resource_id == resource_id
            )
            .order_by(desc(self.model.timestamp))
            .offset(skip)
            .limit(limit)
            .all()
        )


class AlertRepository(BaseRepository[Alert]):
    """Repository for Alert model."""
    
    def __init__(self, db: Session):
        super().__init__(Alert, db)
    
    def get_open_alerts(self, skip: int = 0, limit: int = 100) -> List[Alert]:
        """Get open alerts."""
        return (
            self.db.query(self.model)
            .filter(self.model.status == "open")
            .order_by(desc(self.model.timestamp))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def resolve_alert(self, id: str, resolved_by: str) -> Optional[Alert]:
        """Mark alert as resolved."""
        alert = self.get_by_id(id)
        if alert:
            alert.status = "resolved"
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = resolved_by
            self.db.commit()
            self.db.refresh(alert)
        return alert


class SystemMetricRepository(BaseRepository[SystemMetric]):
    """Repository for SystemMetric model."""
    
    def __init__(self, db: Session):
        super().__init__(SystemMetric, db)
    
    def get_metrics_by_name(self, metric_name: str, skip: int = 0, limit: int = 100) -> List[SystemMetric]:
        """Get metrics by name."""
        return (
            self.db.query(self.model)
            .filter(self.model.metric_name == metric_name)
            .order_by(desc(self.model.timestamp))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def create_batch(self, metrics: List[Dict[str, Any]]) -> List[SystemMetric]:
        """Batch create system metrics."""
        db_metrics = [self.model(**metric) for metric in metrics]
        self.db.add_all(db_metrics)
        self.db.commit()
        return db_metrics


class TagRepository(BaseRepository[Tag]):
    """Repository for Tag model."""
    
    def __init__(self, db: Session):
        super().__init__(Tag, db)
    
    def get_by_name(self, name: str) -> Optional[Tag]:
        """Get tag by name."""
        return self.db.query(self.model).filter(self.model.name == name).first()
    
    def get_or_create(self, name: str) -> Tag:
        """Get tag by name or create if it doesn't exist."""
        tag = self.get_by_name(name)
        if not tag:
            tag = self.create({"name": name})
        return tag
