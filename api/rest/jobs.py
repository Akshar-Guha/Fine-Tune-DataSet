"""Job management REST endpoints."""
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, status
from sqlalchemy.orm import Session
from modelops.db.database import get_db
from modelops.db.repository import JobRepository
from pydantic import BaseModel

from api.auth.permissions import (
    get_current_user,
    Permission
)


router = APIRouter()


class JobType(str, Enum):
    """Job types."""
    SFT_TRAINING = "sft_training"
    QUANTIZATION = "quantization"
    RAG_INDEXING = "rag_indexing"
    RLHF = "rlhf"
    EVALUATION = "evaluation"


class JobStatus(str, Enum):
    """Job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCreate(BaseModel):
    """Job creation request."""
    name: str
    job_type: JobType
    config: Dict[str, Any]
    dataset_id: str
    base_model: Optional[str] = None
    priority: int = 0
    workflow_id: Optional[str] = None


class JobResponse(BaseModel):
    """Job response."""
    job_id: str
    name: str
    job_type: JobType
    status: JobStatus
    config: Dict[str, Any]
    dataset_id: str
    base_model: Optional[str]
    priority: int
    workflow_id: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    updated_at: Optional[datetime]
    duration_seconds: Optional[int]
    error: Optional[str]
    metrics: Optional[Dict[str, Any]]
    artifacts: Optional[Dict[str, Any]]


class JobList(BaseModel):
    """List of jobs."""
    jobs: List[JobResponse]
    total: int


@router.post(
    "/",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_job(
    job: JobCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Submit a new job and persist it in the database."""
    # Check permissions
    if Permission.JOB_SUBMIT.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        repo = JobRepository(db)
        job_dict = job.dict()
        
        # Add creator info
        job_dict["created_by"] = current_user.get("username", "system")
        
        # Submit job to temporal
        db_job = await repo.create_job(job_dict)
        
        # Calculate duration if possible
        duration_seconds = None
        if db_job.completed_at and db_job.started_at:
            duration_seconds = int((db_job.completed_at - db_job.started_at).total_seconds())
            
        return JobResponse(
            job_id=db_job.id,
            name=db_job.name,
            job_type=db_job.job_type,
            status=db_job.status,
            config=db_job.config,
            dataset_id=db_job.dataset_id,
            base_model=db_job.base_model,
            priority=db_job.priority,
            workflow_id=db_job.workflow_id,
            created_at=db_job.created_at,
            started_at=db_job.started_at,
            completed_at=db_job.completed_at,
            updated_at=db_job.updated_at,
            duration_seconds=duration_seconds,
            error=db_job.error,
            metrics=db_job.metrics,
            artifacts=db_job.artifacts
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}"
        )


@router.get("/", response_model=JobList)
async def list_jobs(
    job_type: Optional[JobType] = None,
    status_filter: Optional[JobStatus] = Query(None, alias="status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """List all jobs with optional filters."""
    if Permission.JOB_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    try:
        repo = JobRepository(db)
        jobs, total = repo.list_jobs(
            job_type=job_type.value if job_type else None,
            status=status_filter,
            skip=skip,
            limit=limit,
        )
        job_responses = [
            JobResponse(
                job_id=j.id,
                name=j.name,
                job_type=j.job_type,
                status=j.status,
                config=j.config,
                dataset_id=j.dataset_id,
                base_model=j.base_model,
                priority=j.priority,
                workflow_id=j.workflow_id,
                created_at=j.created_at,
                started_at=j.started_at,
                completed_at=j.completed_at,
                updated_at=j.updated_at,
                duration_seconds=None,
                error=j.error,
                metrics=j.metrics,
                artifacts=j.artifacts
            )
            for j in jobs
        ]
        return JobList(jobs=job_responses, total=total)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}"
        )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get job details by ID."""
    if Permission.JOB_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    try:
        repo = JobRepository(db)
        job = repo.get_by_id(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        return JobResponse(
            job_id=job.id,
            name=job.name,
            job_type=job.job_type,
            status=job.status,
            config=job.config,
            dataset_id=job.dataset_id,
            base_model=job.base_model,
            priority=job.priority,
            workflow_id=job.workflow_id,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            updated_at=job.updated_at,
            duration_seconds=None,
            error=job.error,
            metrics=job.metrics,
            artifacts=job.artifacts
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job: {str(e)}"
        )


@router.post("/{job_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Cancel a running job."""
    # Check permissions
    if Permission.JOB_CANCEL.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        repo = JobRepository(db)
        cancelled_job = await repo.cancel_job(job_id)
        if not cancelled_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        return {"message": f"Job {job_id} cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )


@router.get("/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    tail: int = Query(100, ge=1, le=10000),
    current_user: dict = Depends(get_current_user)
):
    """Get job logs."""
    # Check permissions
    if Permission.JOB_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        # TODO: Implement actual log retrieval logic
        return {"logs": []}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get logs: {str(e)}"
        )
