"""Artifact management REST endpoints."""
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel

from api.auth.permissions import (
    get_current_user,
    require_permissions,
    Permission
)
from artifacts.schemas.base import (
    ArtifactType,
    GovernanceStatus,
    ArtifactManifest
)


router = APIRouter()


class ArtifactSearch(BaseModel):
    """Artifact search query."""
    artifact_type: Optional[ArtifactType] = None
    status: Optional[GovernanceStatus] = None
    tags: Optional[List[str]] = None
    base_model: Optional[str] = None


class ArtifactListResponse(BaseModel):
    """List of artifacts."""
    artifacts: List[ArtifactManifest]
    total: int


class PromoteRequest(BaseModel):
    """Artifact promotion request."""
    target_status: GovernanceStatus
    approver: str
    notes: Optional[str] = None


@router.post(
    "/",
    response_model=ArtifactManifest,
    status_code=status.HTTP_201_CREATED
)
async def register_artifact(
    manifest: ArtifactManifest,
    current_user: dict = Depends(get_current_user)
):
    """Register a new artifact."""
    # Check permissions
    if Permission.ARTIFACT_WRITE.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        from artifacts.registry.manager import ArtifactRegistry
        registry = ArtifactRegistry()
        artifact_id = registry.register(manifest)

        # Return registered manifest
        return registry.get(artifact_id)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register artifact: {str(e)}"
        )


@router.get("/", response_model=ArtifactListResponse)
async def list_artifacts(
    artifact_type: Optional[ArtifactType] = None,
    status_filter: Optional[GovernanceStatus] = Query(None, alias="status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: dict = Depends(get_current_user)
):
    """List artifacts."""
    # Check permissions
    if Permission.ARTIFACT_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        from artifacts.registry.manager import ArtifactRegistry
        registry = ArtifactRegistry()

        # Search with filters
        results = registry.search(
            artifact_type=artifact_type.value if artifact_type else None,
            status=status_filter.value if status_filter else None,
            limit=limit
        )

        return ArtifactListResponse(
            artifacts=results,
            total=len(results)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list artifacts: {str(e)}"
        )


@router.get("/{artifact_id}", response_model=ArtifactManifest)
async def get_artifact(
    artifact_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get artifact details."""
    # Check permissions
    if Permission.ARTIFACT_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        from artifacts.registry.manager import ArtifactRegistry
        registry = ArtifactRegistry()
        manifest = registry.get(artifact_id)

        if not manifest:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Artifact {artifact_id} not found"
            )

        return manifest

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get artifact: {str(e)}"
        )


@router.post("/{artifact_id}/promote", response_model=ArtifactManifest)
async def promote_artifact(
    artifact_id: str,
    request: PromoteRequest,
    current_user: dict = Depends(get_current_user)
):
    """Promote artifact to next governance stage."""
    # Check permissions
    if Permission.ARTIFACT_PROMOTE.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        from artifacts.registry.manager import ArtifactRegistry
        registry = ArtifactRegistry()
        registry.promote(
            artifact_id,
            request.target_status,
            request.approver
        )

        return registry.get(artifact_id)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to promote artifact: {str(e)}"
        )


@router.get("/{artifact_id}/lineage")
async def get_artifact_lineage(
    artifact_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get artifact lineage (upstream and downstream)."""
    # Check permissions
    if Permission.ARTIFACT_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        from artifacts.registry.manager import ArtifactRegistry
        registry = ArtifactRegistry()
        lineage = registry.get_lineage(artifact_id)

        return lineage

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get lineage: {str(e)}"
        )


@router.get("/{artifact_id}/versions")
async def get_artifact_versions(
    artifact_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all versions of an artifact."""
    # Check permissions
    if Permission.ARTIFACT_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        from artifacts.registry.manager import ArtifactRegistry
        registry = ArtifactRegistry()
        versions = registry.list_versions(artifact_id)

        return {"artifact_id": artifact_id, "versions": versions}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get versions: {str(e)}"
        )


@router.delete("/{artifact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_artifact(
    artifact_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete an artifact."""
    # Check permissions
    if Permission.ARTIFACT_DELETE.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        # TODO: Implement actual deletion logic
        pass

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete artifact: {str(e)}"
        )
