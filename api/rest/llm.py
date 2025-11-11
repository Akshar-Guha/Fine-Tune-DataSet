"""LLM model endpoints."""
from typing import List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, status
from sqlalchemy.orm import Session

from modelops.db.database import get_db
from modelops.db.models import LLMVersionStatus
from modelops.db.repository import LLMRepository, LLMVersionRepository
from api.rest.models import (
    LLM,
    LLMCreate,
    LLMUpdate,
    LLMWithVersions,
    LLMVersion,
    LLMVersionCreate,
    LLMVersionUpdate
)

from api.auth.permissions import (
    get_current_user,
    Permission
)

router = APIRouter()


@router.post("/", response_model=LLM, status_code=status.HTTP_201_CREATED)
async def create_llm(
    llm: LLMCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new LLM model."""
    # Check permissions
    if Permission.MODEL_CREATE.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        repo = LLMRepository(db)
        existing = repo.get_by_name(llm.name)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"LLM with name '{llm.name}' already exists"
            )
        
        db_llm = repo.create(llm.dict())
        return db_llm
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create LLM: {str(e)}"
        )


@router.get("/", response_model=List[LLM])
async def list_llms(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """List all LLM models."""
    if Permission.MODEL_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        repo = LLMRepository(db)
        return repo.get_all(skip, limit)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list LLMs: {str(e)}"
        )


@router.get("/{llm_id}", response_model=LLMWithVersions)
async def get_llm(
    llm_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get LLM details by ID."""
    if Permission.MODEL_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        repo = LLMRepository(db)
        llm = repo.get_with_versions(llm_id)
        if not llm:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM {llm_id} not found"
            )
        return llm
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get LLM: {str(e)}"
        )


@router.put("/{llm_id}", response_model=LLM)
async def update_llm(
    llm_id: str,
    llm_update: LLMUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update LLM details."""
    if Permission.MODEL_UPDATE.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        repo = LLMRepository(db)
        existing = repo.get_by_id(llm_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM {llm_id} not found"
            )
        
        # Check for name conflicts if name is being updated
        if llm_update.name and llm_update.name != existing.name:
            name_exists = repo.get_by_name(llm_update.name)
            if name_exists:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"LLM with name '{llm_update.name}' already exists"
                )
        
        updated_llm = repo.update(llm_id, llm_update.dict(exclude_unset=True))
        return updated_llm
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update LLM: {str(e)}"
        )


@router.delete("/{llm_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_llm(
    llm_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete an LLM."""
    if Permission.MODEL_DELETE.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        repo = LLMRepository(db)
        existing = repo.get_by_id(llm_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM {llm_id} not found"
            )
        
        # Check if LLM has any versions
        version_repo = LLMVersionRepository(db)
        versions = version_repo.get_versions_by_llm(llm_id)
        if versions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete LLM with existing versions"
            )
        
        repo.delete(llm_id)
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete LLM: {str(e)}"
        )


# LLM Version endpoints
@router.post("/{llm_id}/versions", response_model=LLMVersion)
async def create_llm_version(
    llm_id: str,
    version: LLMVersionCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new version of an LLM model."""
    if Permission.MODEL_CREATE.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        # Check if LLM exists
        llm_repo = LLMRepository(db)
        llm = llm_repo.get_by_id(llm_id)
        if not llm:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM {llm_id} not found"
            )
        
        # Create version
        version_repo = LLMVersionRepository(db)
        
        # Check for version conflicts
        existing_version = version_repo.get_by_llm_and_version(llm_id, version.version)
        if existing_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Version {version.version} already exists for LLM {llm_id}"
            )
        
        # Override llm_id from path
        version_data = version.dict()
        version_data["llm_id"] = llm_id
        
        # Add created_by from current user if not provided
        if "created_by" not in version_data or not version_data["created_by"]:
            version_data["created_by"] = current_user.get("user_id")
            
        db_version = version_repo.create(version_data)
        
        return db_version
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create LLM version: {str(e)}"
        )


@router.get("/{llm_id}/versions", response_model=List[LLMVersion])
async def list_llm_versions(
    llm_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """List all versions of an LLM model."""
    if Permission.MODEL_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        # Check if LLM exists
        llm_repo = LLMRepository(db)
        llm = llm_repo.get_by_id(llm_id)
        if not llm:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM {llm_id} not found"
            )
        
        # Get versions
        version_repo = LLMVersionRepository(db)
        return version_repo.get_versions_by_llm(llm_id, skip, limit)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list LLM versions: {str(e)}"
        )


@router.get("/{llm_id}/versions/latest", response_model=LLMVersion)
async def get_latest_llm_version(
    llm_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get the latest version of an LLM model."""
    if Permission.MODEL_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        # Check if LLM exists
        llm_repo = LLMRepository(db)
        llm = llm_repo.get_by_id(llm_id)
        if not llm:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LLM {llm_id} not found"
            )
        
        # Get latest version
        version_repo = LLMVersionRepository(db)
        version = version_repo.get_latest_version(llm_id)
        if not version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No versions found for LLM {llm_id}"
            )
        
        return version
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get latest LLM version: {str(e)}"
        )


@router.get("/{llm_id}/versions/{version}", response_model=LLMVersion)
async def get_llm_version(
    llm_id: str,
    version: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get a specific version of an LLM model."""
    if Permission.MODEL_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        version_repo = LLMVersionRepository(db)
        llm_version = version_repo.get_by_llm_and_version(llm_id, version)
        if not llm_version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Version {version} not found for LLM {llm_id}"
            )
        
        return llm_version
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get LLM version: {str(e)}"
        )


@router.put("/{llm_id}/versions/{version_id}", response_model=LLMVersion)
async def update_llm_version(
    llm_id: str,
    version_id: str,
    version_update: LLMVersionUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update a specific version of an LLM model."""
    if Permission.MODEL_UPDATE.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        version_repo = LLMVersionRepository(db)
        version = version_repo.get_by_id(version_id)
        
        if not version or version.llm_id != llm_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Version {version_id} not found for LLM {llm_id}"
            )
        
        updated_version = version_repo.update(version_id, version_update.dict(exclude_unset=True))
        return updated_version
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update LLM version: {str(e)}"
        )


@router.post("/{llm_id}/versions/{version_id}/promote")
async def promote_llm_version(
    llm_id: str,
    version_id: str,
    stage: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Promote LLM version to a new stage (e.g., from DRAFT to COMPLETED)."""
    if Permission.MODEL_UPDATE.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        # Validate stage
        try:
            target_status = LLMVersionStatus(stage.lower())
        except ValueError:
            valid_statuses = [s.value for s in LLMVersionStatus]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid stage '{stage}'. Must be one of: {', '.join(valid_statuses)}"
            )
        
        # Check version exists for this LLM
        version_repo = LLMVersionRepository(db)
        version = version_repo.get_by_id(version_id)
        if not version or version.llm_id != llm_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Version {version_id} not found for LLM {llm_id}"
            )
        
        # Validate promotion
        if target_status == LLMVersionStatus.DEPLOYED:
            if version.status != LLMVersionStatus.COMPLETED:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Can only deploy models that are in COMPLETED status"
                )
        
        updates = {"status": target_status}
        
        # Set completed_at if moving to COMPLETED status
        if target_status == LLMVersionStatus.COMPLETED and not version.completed_at:
            updates["completed_at"] = datetime.utcnow()
        
        updated_version = version_repo.update(version_id, updates)
        
        return {"message": f"Version {version.version} promoted to {stage} successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to promote LLM version: {str(e)}"
        )
