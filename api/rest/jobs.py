"""Job management REST endpoints."""
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel

from api.auth.permissions import (
    get_current_user,
    require_permissions,
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


class JobResponse(BaseModel):
    """Job response."""
    job_id: str
    name: str
    job_type: JobType
    status: JobStatus
    config: Dict[str, Any]
    dataset_id: str
    base_model: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[int]
    error: Optional[str]
    metrics: Optional[Dict[str, Any]]
    artifacts: List[str]


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
    current_user: dict = Depends(get_current_user)
):
    """Submit a new job."""
    # Check permissions
    if Permission.JOB_SUBMIT.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        # TODO: Implement actual job submission to Temporal
        # For now, return mock response
        import uuid
        job_id = str(uuid.uuid4())

        return JobResponse(
            job_id=job_id,
            name=job.name,
            job_type=job.job_type,
            status=JobStatus.PENDING,
            config=job.config,
            dataset_id=job.dataset_id,
            base_model=job.base_model,
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None,
            duration_seconds=None,
            error=None,
            metrics=None,
            artifacts=[]
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
    current_user: dict = Depends(get_current_user)
):
    """List all jobs."""
    # Check permissions
    if Permission.JOB_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        # TODO: Implement actual job listing logic
        return JobList(jobs=[], total=0)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list jobs: {str(e)}"
        )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get job details."""
    # Check permissions
    if Permission.JOB_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        # TODO: Implement actual job retrieval logic
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
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
        # TODO: Implement actual job cancellation logic
        return {"message": f"Job {job_id} cancelled"}

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
