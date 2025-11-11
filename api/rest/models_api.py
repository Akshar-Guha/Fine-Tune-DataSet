"""Models API endpoints for frontend compatibility."""
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Query, status
from sqlalchemy.orm import Session

from modelops.db.database import get_db
from modelops.db.repository import LLMRepository, LLMVersionRepository
from api.rest.models import ModelMetrics

from api.auth.permissions import (
    get_current_user,
    Permission
)

router = APIRouter()


@router.get("/", response_model=Dict[str, Any])
async def list_models(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """List all models with optional filters."""
    if Permission.MODEL_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        llm_repo = LLMRepository(db)
        llms = llm_repo.get_all(skip, limit)

        models = []
        for llm in llms:
            # Get latest version for status
            version_repo = LLMVersionRepository(db)
            latest_version = version_repo.get_latest_version(llm.id)

            model_dict = {
                "id": llm.id,
                "name": llm.name,
                "description": llm.description,
                "base_model": llm.base_model,
                "created_at": llm.created_at.isoformat(),
                "updated_at": llm.updated_at.isoformat(),
                "versions": []
            }

            if latest_version:
                model_dict["versions"] = [{
                    "id": latest_version.id,
                    "version": latest_version.version,
                    "llm_id": latest_version.llm_id,
                    "status": latest_version.status,
                    "created_by": latest_version.created_by,
                    "created_at": latest_version.created_at.isoformat(),
                    "updated_at": latest_version.updated_at.isoformat(),
                    "completed_at": (
                        latest_version.completed_at.isoformat()
                        if latest_version.completed_at else None
                    ),
                    "training_dataset_version_id": (
                        latest_version.training_dataset_version_id
                    ),
                    "validation_dataset_version_id": (
                        latest_version.validation_dataset_version_id
                    ),
                    "model_size": latest_version.model_size,
                    "checkpoint_path": latest_version.checkpoint_path,
                    "config": latest_version.config,
                    "training_metrics": latest_version.training_metrics,
                    "validation_metrics": latest_version.validation_metrics,
                    "parameters": [],  # TODO: populate if needed
                    "tags": []  # TODO: populate if needed
                }]

            models.append(model_dict)

        return {"models": models, "total": len(models)}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/{model_id}", response_model=Dict[str, Any])
async def get_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get model details by ID."""
    if Permission.MODEL_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        llm_repo = LLMRepository(db)
        llm = llm_repo.get_with_versions(model_id)
        if not llm:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )

        # Convert to frontend expected format
        model_dict = {
            "id": llm.id,
            "name": llm.name,
            "description": llm.description,
            "base_model": llm.base_model,
            "created_at": llm.created_at.isoformat(),
            "updated_at": llm.updated_at.isoformat(),
            "versions": []
        }

        for version in llm.versions:
            version_dict = {
                "id": version.id,
                "version": version.version,
                "llm_id": version.llm_id,
                "status": version.status,
                "created_by": version.created_by,
                "created_at": version.created_at.isoformat(),
                "updated_at": version.updated_at.isoformat(),
                "completed_at": (
                    version.completed_at.isoformat()
                    if version.completed_at else None
                ),
                "training_dataset_version_id": (
                    version.training_dataset_version_id
                ),
                "validation_dataset_version_id": (
                    version.validation_dataset_version_id
                ),
                "model_size": version.model_size,
                "checkpoint_path": version.checkpoint_path,
                "config": version.config,
                "training_metrics": version.training_metrics,
                "validation_metrics": version.validation_metrics,
                "parameters": [],  # TODO: populate
                "tags": []  # TODO: populate
            }
            model_dict["versions"].append(version_dict)

        return model_dict

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model: {str(e)}"
        )


@router.get("/{model_id}/metrics", response_model=ModelMetrics)
async def get_model_metrics(
    model_id: str,
    time_range: Optional[str] = Query("7d"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get model metrics."""
    if Permission.MODEL_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        # TODO: Implement actual metrics calculation from training logs
        # For now, return mock data
        return ModelMetrics(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            history=[],
            confusion_matrix={
                "labels": ["class1", "class2"],
                "values": [[50, 5], [10, 35]]
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model metrics: {str(e)}"
        )


@router.get("/{model_id}/versions", response_model=List[Dict[str, Any]])
async def get_model_versions(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get model versions."""
    if Permission.MODEL_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        version_repo = LLMVersionRepository(db)
        versions = version_repo.get_versions_by_llm(model_id)

        return [{
            "id": v.id,
            "version": v.version,
            "llm_id": v.llm_id,
            "status": v.status,
            "created_by": v.created_by,
            "created_at": v.created_at.isoformat(),
            "updated_at": v.updated_at.isoformat(),
            "completed_at": (
                v.completed_at.isoformat() if v.completed_at else None
            ),
            "training_dataset_version_id": v.training_dataset_version_id,
            "validation_dataset_version_id": v.validation_dataset_version_id,
            "model_size": v.model_size,
            "checkpoint_path": v.checkpoint_path,
            "config": v.config,
            "training_metrics": v.training_metrics,
            "validation_metrics": v.validation_metrics,
            "parameters": [],  # TODO: populate
            "tags": []  # TODO: populate
        } for v in versions]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model versions: {str(e)}"
        )


@router.post("/compare", response_model=Dict[str, Any])
async def compare_models(
    request: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Compare models."""
    if Permission.MODEL_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        # TODO: Implement actual model comparison
        return {"comparison": "Not implemented yet"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare models: {str(e)}"
        )


@router.post("/{model_id}/deploy", response_model=Dict[str, Any])
async def deploy_model(
    model_id: str,
    config: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Deploy model."""
    if Permission.DEPLOY_CREATE.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        # TODO: Implement actual deployment
        return {"deployment_id": "mock_deployment_id", "status": "deployed"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deploy model: {str(e)}"
        )


@router.post("/{model_id}/promote", response_model=Dict[str, Any])
async def promote_model(
    model_id: str,
    request: Dict[str, str],
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Promote model to a stage."""
    if Permission.ARTIFACT_PROMOTE.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        # TODO: Implement actual promotion
        return {
            "status": "promoted",
            "stage": request.get("stage", "production")
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to promote model: {str(e)}"
        )
