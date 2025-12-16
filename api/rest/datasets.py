"""Dataset management REST endpoints."""
import json
from dataclasses import asdict
from typing import Optional, List, Dict, Any

from datetime import datetime

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from data_management.quality.pipeline import DatasetIngestionPipeline
from data_management.quality.pipeline_config import AutoLabelingRule
from services.dataset_registry import DatasetRegistry
from services.storage.dataset_manager import DatasetManager


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
    quality_status: Optional[str] = None
    quality_score: Optional[float] = None
    validation_report_path: Optional[str] = None
    last_validated: Optional[datetime] = None
    created_at: datetime


class DatasetList(BaseModel):
    """List of datasets."""
    datasets: List[DatasetResponse]
    total: int


class DatasetValidationSummary(BaseModel):
    """Validation summary for uploads."""

    status: str
    quality_score: Optional[float] = None
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    quality_report_path: str


class DatasetUploadResponse(BaseModel):
    """Upload response payload."""

    dataset_id: str
    dataset_name: str
    quality_passed: bool
    validation: DatasetValidationSummary
    registry_path: Optional[str] = None
    processed_file_path: Optional[str] = None


class DatasetValidationRecord(BaseModel):
    """Single validation record."""

    dataset_id: str
    dataset_name: Optional[str] = None
    status: str
    quality_score: Optional[float] = None
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    quality_report_path: str
    created_at: datetime


class ProcessLocalDatasetRequest(BaseModel):
    """Request to process an existing local dataset."""
    dataset_id: str
    text_column: Optional[str] = None
    label_column: Optional[str] = None
    enable_auto_labeling: bool = True
    auto_labeling_rules: Optional[List[Dict[str, Any]]] = None
    quality_score_threshold: int = 80
    strip_text: bool = True
    drop_missing_text: bool = True
    drop_missing_label: bool = True
    lowercase_labels: bool = True
    drop_duplicates: bool = True


class LocalDatasetSummary(BaseModel):
    """Summary of locally stored datasets."""

    dataset_id: str
    local_path: str
    num_rows: int
    num_columns: int
    columns: List[str]
    size_mb: float
    splits: List[str]
    downloaded: bool


class DatasetDownloadRequest(BaseModel):
    """Dataset download request."""

    dataset_id: str
    split: Optional[str] = None
    subset: Optional[str] = None
    force: bool = False
def get_dataset_manager() -> DatasetManager:
    """Provide DatasetManager instance."""
    return DatasetManager()


def get_dataset_pipeline() -> DatasetIngestionPipeline:
    """Provide DatasetIngestionPipeline instance."""

    return DatasetIngestionPipeline()


def get_dataset_registry() -> DatasetRegistry:
    """Provide DatasetRegistry instance."""

    return DatasetRegistry()


def _build_dataset_response(metadata: Dict[str, Any]) -> DatasetResponse:
    schema_entries = metadata.get("schema") or []
    return DatasetResponse(
        id=metadata["id"],
        name=metadata["name"],
        description=metadata.get("description"),
        file_path=metadata["file_path"],
        format=metadata["format"],
        rows=metadata["rows"],
        columns=metadata["columns"],
        size_bytes=metadata["size_bytes"],
        schema=schema_entries,
        quality_status=metadata.get("quality_status"),
        quality_score=metadata.get("quality_score"),
        validation_report_path=metadata.get("validation_report_path"),
        last_validated=metadata.get("last_validated"),
        created_at=metadata["created_at"],
    )


@router.post(
    "/",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_dataset(
    dataset: DatasetCreate,
    manager: DatasetManager = Depends(get_dataset_manager)
):
    """Create a new dataset."""
    # Check permissions

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

        return _build_dataset_response(metadata)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create dataset: {str(e)}"
        )


@router.get("/", response_model=DatasetList)
async def list_datasets(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    manager: DatasetManager = Depends(get_dataset_manager)
):
    """List all datasets."""
    # Check permissions

    try:
        datasets = manager.list_datasets()
        sliced = datasets[skip: skip + limit]
        items = [_build_dataset_response(record) for record in sliced]
        return DatasetList(datasets=items, total=len(datasets))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}"
        )


@router.get("/search", response_model=List[Dict[str, Any]])
async def search_datasets_api(
    query: str = Query(..., description="Search query for datasets"),
    task: Optional[str] = Query("text-generation", description="Dataset task filter"),
    limit: int = Query(20, ge=1, le=50, description="Maximum number of results"),
    registry: DatasetRegistry = Depends(get_dataset_registry),
):
    """Search for datasets on HuggingFace Hub."""

    try:
        results = registry.search_datasets(
            query=query,
            task=task,
            limit=limit
        )
        return results
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search datasets: {exc}",
        ) from exc


@router.post("/download", response_model=Dict[str, str])
async def download_dataset_api(
    request: DatasetDownloadRequest,
    registry: DatasetRegistry = Depends(get_dataset_registry),
):
    """Download a dataset from HuggingFace Hub."""

    try:
        local_path = registry.download_dataset(
            dataset_id=request.dataset_id,
            split=request.split,
            subset=request.subset,
            force=request.force
        )
        return {"path": local_path}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download dataset: {exc}",
        ) from exc


@router.get("/local", response_model=List[LocalDatasetSummary])
async def list_local_datasets_api(
    registry: DatasetRegistry = Depends(get_dataset_registry),
):
    """List datasets stored locally on disk."""

    try:
        datasets = registry.list_local_datasets()
        return [LocalDatasetSummary(**asdict(dataset)) for dataset in datasets]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list local datasets: {exc}",
        ) from exc


@router.get("/quality-report", response_model=Dict[str, Any])
async def get_quality_report(
    report_path: str = Query(..., description="Path to a quality_report.json returned by the pipeline"),
    registry: DatasetRegistry = Depends(get_dataset_registry),
):
    """Return the JSON content of a stored quality report."""

    try:
        return registry.load_quality_report(report_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quality report not found",
        ) from exc
    except ValueError as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get("/{dataset_id:path}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    manager: DatasetManager = Depends(get_dataset_manager)
):
    """Get dataset details."""
    # Check permissions

    try:
        dataset = manager.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )

        return _build_dataset_response(dataset)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dataset: {str(e)}"
        )


@router.delete("/{dataset_id:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str,
    manager: DatasetManager = Depends(get_dataset_manager)
):
    """Delete a dataset."""
    # Check permissions

    try:
        deleted = manager.delete_dataset(dataset_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete dataset: {str(e)}"
        )


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_dataset(
    *,
    file: UploadFile = File(...),
    dataset_name: Optional[str] = Form(None),
    text_column: Optional[str] = Form(None),
    label_column: Optional[str] = Form(None),
    pipeline_config: Optional[str] = Form(None),
    manager: DatasetManager = Depends(get_dataset_manager),
    pipeline: DatasetIngestionPipeline = Depends(get_dataset_pipeline),
):
    """Upload, validate, and optionally register a new dataset."""

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must include a filename",
        )

    dataset_identifier = pipeline.generate_dataset_id(
        dataset_name=dataset_name,
        upload_filename=file.filename,
    )

    existing_dataset = manager.get_dataset(dataset_identifier)

    overrides: Dict[str, Any] = {}
    if pipeline_config:
        try:
            overrides_raw = json.loads(pipeline_config)
        except json.JSONDecodeError as exc:  # noqa: PERF203
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid pipeline configuration payload",
            ) from exc

        if not isinstance(overrides_raw, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Pipeline configuration must be an object",
            )
        overrides = overrides_raw

    auto_labeling_rules_payload = overrides.get("auto_labeling_rules")
    auto_labeling_rules = None
    if isinstance(auto_labeling_rules_payload, list):
        auto_labeling_rules = []
        for rule in auto_labeling_rules_payload:
            if not isinstance(rule, dict):
                continue
            label = rule.get("label")
            keywords = rule.get("keywords", [])
            if not label:
                continue
            auto_labeling_rules.append(
                AutoLabelingRule(label=label, keywords=list(keywords))
            )

    try:
        print(f"DEBUG: Starting pipeline run for file: {file.filename}")
        result = pipeline.run(
            dataset_name=dataset_name,
            upload_filename=file.filename,
            file_obj=file.file,
            text_column=text_column,
            label_column=label_column,
            dataset_id=dataset_identifier,
            enable_auto_labeling=overrides.get("enable_auto_labeling"),
            auto_labeling_rules=auto_labeling_rules,
            quality_score_threshold=overrides.get("quality_score_threshold"),
            strip_text=overrides.get("strip_text"),
            drop_missing_text=overrides.get("drop_missing_text"),
            drop_missing_label=overrides.get("drop_missing_label"),
            lowercase_labels=overrides.get("lowercase_labels"),
            drop_duplicates=overrides.get("drop_duplicates", True),
        )
        print(f"DEBUG: Pipeline run completed successfully")
    except ValueError as exc:
        print(f"DEBUG: ValueError in pipeline: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"DEBUG: Exception in pipeline: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {exc}",
        ) from exc

    quality_score = result.quality_report.get("quality_score")
    status_label = "passed" if result.quality_passed else "failed"

    manager.record_validation_result(
        dataset_id=result.dataset_id,
        dataset_name=result.dataset_name,
        status=status_label,
        quality_score=quality_score,
        report_path=str(result.quality_report_path),
        issues=result.issues,
    )

    registry_path = None
    processed_file_path = None

    has_processed_outputs = bool(
        result.processed_file_path and result.registry_path
    )

    if result.quality_passed and has_processed_outputs:
        processed_file_path = str(result.processed_file_path)
        registry_path = str(result.registry_path)
        try:
            manager.add_dataset(
                dataset_id=result.dataset_id,
                name=result.dataset_name,
                file_path=processed_file_path,
                description="",
                format="parquet",
                quality_status="passed",
                quality_score=quality_score,
                validation_report_path=str(result.quality_report_path),
                last_validated=datetime.utcnow(),
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Processed dataset missing: {exc}",
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to persist dataset metadata: {exc}",
            ) from exc
    else:
        # Update quality status if dataset already exists in registry.
        if existing_dataset:
            manager.update_dataset_quality(
                dataset_id=result.dataset_id,
                status=status_label,
                quality_score=quality_score,
                validation_report_path=str(result.quality_report_path),
                last_validated=datetime.utcnow(),
            )

    validation_summary = DatasetValidationSummary(
        status=status_label,
        quality_score=quality_score,
        issues=result.issues,
        quality_report_path=str(result.quality_report_path),
    )

    return DatasetUploadResponse(
        dataset_id=result.dataset_id,
        dataset_name=result.dataset_name,
        quality_passed=result.quality_passed,
        validation=validation_summary,
        registry_path=registry_path,
        processed_file_path=processed_file_path,
    )


@router.get(
    "/validation/recent",
    response_model=List[DatasetValidationRecord],
    status_code=status.HTTP_200_OK,
)
async def list_validation_results(
    limit: int = Query(20, ge=1, le=100),
    manager: DatasetManager = Depends(get_dataset_manager),
):
    """Return recent validation outcomes for datasets."""

    records = manager.list_validation_results(limit=limit)
    return [
        DatasetValidationRecord(
            dataset_id=item["dataset_id"],
            dataset_name=item.get("dataset_name"),
            status=item["status"],
            quality_score=item.get("quality_score"),
            issues=item.get("issues", []),
            quality_report_path=item["report_path"],
            created_at=item["created_at"],
        )
        for item in records
    ]


@router.get(
    "/{dataset_id:path}/validation/latest",
    response_model=DatasetValidationRecord,
    status_code=status.HTTP_200_OK,
)
async def get_latest_validation(
    dataset_id: str,
    manager: DatasetManager = Depends(get_dataset_manager),
):
    """Return latest validation record for a dataset."""

    record = manager.get_latest_validation_result(dataset_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No validation results found for {dataset_id}",
        )

    return DatasetValidationRecord(
        dataset_id=record["dataset_id"],
        dataset_name=record.get("dataset_name"),
        status=record["status"],
        quality_score=record.get("quality_score"),
        issues=record.get("issues", []),
        quality_report_path=record["report_path"],
        created_at=record["created_at"],
    )


@router.get(
    "/{dataset_id:path}/validation/history",
    response_model=List[DatasetValidationRecord],
    status_code=status.HTTP_200_OK,
)
async def get_validation_history(
    dataset_id: str,
    limit: Optional[int] = Query(None, ge=1, le=200),
    manager: DatasetManager = Depends(get_dataset_manager),
):
    """Return validation history for a dataset."""

    records = manager.get_validation_history(dataset_id, limit=limit)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No validation results found for {dataset_id}",
        )

    return [
        DatasetValidationRecord(
            dataset_id=item["dataset_id"],
            dataset_name=item.get("dataset_name"),
            status=item["status"],
            quality_score=item.get("quality_score"),
            issues=item.get("issues", []),
            quality_report_path=item["report_path"],
            created_at=item["created_at"],
        )
        for item in records
    ]


@router.post(
    "/process/{dataset_id:path}",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_local_dataset(
    dataset_id: str,
    request: ProcessLocalDatasetRequest,
    manager: DatasetManager = Depends(get_dataset_manager),
    pipeline: DatasetIngestionPipeline = Depends(get_dataset_pipeline),
    registry: DatasetRegistry = Depends(get_dataset_registry),
):
    """Process an existing local dataset through the pipeline."""

    # Get the local dataset info
    local_datasets = registry.list_local_datasets()
    local_dataset = next((d for d in local_datasets if d.dataset_id == dataset_id), None)

    if not local_dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Local dataset {dataset_id} not found"
        )

    # Convert auto_labeling_rules to the expected format
    auto_labeling_rules = None
    if request.auto_labeling_rules:
        auto_labeling_rules = [
            AutoLabelingRule(**rule) for rule in request.auto_labeling_rules
        ]

    # Create a temporary file-like object from the local dataset
    try:
        result = pipeline.run_from_huggingface_dataset(
            dataset_path=local_dataset.local_path,
            dataset_name=dataset_id,
            text_column=request.text_column,
            label_column=request.label_column,
            auto_labeling_rules=auto_labeling_rules,
            enable_auto_labeling=request.enable_auto_labeling,
            quality_score_threshold=request.quality_score_threshold,
            strip_text=request.strip_text,
            drop_missing_text=request.drop_missing_text,
            drop_missing_label=request.drop_missing_label,
            lowercase_labels=request.lowercase_labels,
            dataset_id=dataset_id,
            drop_duplicates=request.drop_duplicates,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {exc}",
        )

    quality_score = result.quality_report.get("quality_score")
    status_label = "passed" if result.quality_passed else "failed"

    manager.record_validation_result(
        dataset_id=result.dataset_id,
        dataset_name=result.dataset_name,
        status=status_label,
        quality_score=quality_score,
        report_path=str(result.quality_report_path),
        issues=result.issues,
    )

    registry_path = None
    processed_file_path = None

    has_processed_outputs = bool(
        result.processed_file_path and result.registry_path
    )

    if result.quality_passed and has_processed_outputs:
        processed_file_path = str(result.processed_file_path)
        registry_path = str(result.registry_path)
        try:
            manager.add_dataset(
                dataset_id=result.dataset_id,
                name=result.dataset_name,
                file_path=processed_file_path,
                description="",
                format="parquet",
                quality_status="passed",
                quality_score=quality_score,
                validation_report_path=str(result.quality_report_path),
                last_validated=datetime.utcnow(),
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Processed dataset missing: {exc}",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to persist dataset metadata: {exc}",
            )
    else:
        # Update quality status if dataset already exists in registry.
        existing_dataset = manager.get_dataset(result.dataset_id)
        if existing_dataset:
            manager.update_dataset_quality(
                dataset_id=result.dataset_id,
                status=status_label,
                quality_score=quality_score,
                validation_report_path=str(result.quality_report_path),
                last_validated=datetime.utcnow(),
            )

    validation_summary = DatasetValidationSummary(
        status=status_label,
        quality_score=quality_score,
        issues=result.issues,
        quality_report_path=str(result.quality_report_path),
    )

    return DatasetUploadResponse(
        dataset_id=result.dataset_id,
        dataset_name=result.dataset_name,
        quality_passed=result.quality_passed,
        validation=validation_summary,
        registry_path=registry_path,
        processed_file_path=processed_file_path,
    )
