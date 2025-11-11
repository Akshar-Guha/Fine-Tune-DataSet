"""Dataset management REST endpoints."""
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel

from services.storage.dataset_manager import DatasetManager

from api.auth.permissions import (
    get_current_user,
    Permission
)


router = APIRouter()


class DatasetCreate(BaseModel):
    """Dataset creation request."""
    name: str
    description: Optional[str] = None
    source_path: str
    create_embeddings: bool = False
    embedding_model: str = "BAAI/bge-small-en-v1.5"


class DatasetResponse(BaseModel):
    """Dataset response."""
    id: str
    name: str
    description: Optional[str]
    file_path: str
    format: str
    rows: int
    columns: int
    size_bytes: int
    schema: List[dict]
    created_at: datetime


class DatasetList(BaseModel):
    """List of datasets."""
    datasets: List[DatasetResponse]
    total: int


def get_dataset_manager() -> DatasetManager:
    """Provide DatasetManager instance."""
    return DatasetManager()


@router.post(
    "/",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_dataset(
    dataset: DatasetCreate,
    current_user: dict = Depends(get_current_user),
    manager: DatasetManager = Depends(get_dataset_manager)
):
    """Create a new dataset."""
    # Check permissions
    if Permission.DATASET_WRITE.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        dataset_id = dataset.name
        metadata = manager.add_dataset(
            dataset_id=dataset_id,
            name=dataset.name,
            file_path=dataset.source_path,
            description=dataset.description or "",
            format="parquet" if dataset.source_path.endswith(".parquet") else (
                "csv" if dataset.source_path.endswith(".csv") else "json"
            )
        )

        return DatasetResponse(**metadata)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create dataset: {str(e)}"
        )


@router.get("/", response_model=DatasetList)
async def list_datasets(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: dict = Depends(get_current_user),
    manager: DatasetManager = Depends(get_dataset_manager)
):
    """List all datasets."""
    # Check permissions
    if Permission.DATASET_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        datasets = manager.list_datasets()
        sliced = datasets[skip: skip + limit]
        items = [DatasetResponse(**record) for record in sliced]
        return DatasetList(datasets=items, total=len(datasets))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}"
        )


@router.get("/{dataset_name}", response_model=DatasetResponse)
async def get_dataset(
    dataset_name: str,
    current_user: dict = Depends(get_current_user),
    manager: DatasetManager = Depends(get_dataset_manager)
):
    """Get dataset details."""
    # Check permissions
    if Permission.DATASET_READ.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        dataset = manager.get_dataset(dataset_name)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_name} not found"
            )

        return DatasetResponse(**dataset)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset: {str(e)}"
        )


@router.delete("/{dataset_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_name: str,
    current_user: dict = Depends(get_current_user),
    manager: DatasetManager = Depends(get_dataset_manager)
):
    """Delete a dataset."""
    # Check permissions
    if Permission.DATASET_DELETE.value not in current_user["permissions"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )

    try:
        deleted = manager.delete_dataset(dataset_name)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_name} not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete dataset: {str(e)}"
        )
