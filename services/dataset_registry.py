"""Dataset Registry Service - Fetch and manage datasets locally."""
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from datasets import load_dataset, DatasetDict, Dataset
from datasets import load_from_disk
from huggingface_hub import HfApi, dataset_info
import pandas as pd


@dataclass
class DatasetInfo:
    """Dataset metadata."""
    dataset_id: str
    local_path: str
    num_rows: int
    num_columns: int
    columns: List[str]
    size_mb: float
    splits: List[str]
    downloaded: bool


class DatasetRegistry:
    """Manage datasets with HuggingFace Hub integration."""
    
    def __init__(self, datasets_dir: str = "./datasets"):
        """Initialize dataset registry.
        
        Args:
            datasets_dir: Local directory for storing datasets
        """
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.api = HfApi()
        
    def search_datasets(
        self,
        query: str = "",
        task: str = "text-generation",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search HuggingFace Hub for datasets.
        
        Args:
            query: Search query
            task: Dataset task
            limit: Maximum results
            
        Returns:
            List of dataset metadata
        """
        try:
            datasets = self.api.list_datasets(
                filter=task if task else None,
                search=query,
                limit=limit
            )
            
            results = []
            for ds in datasets:
                try:
                    info = dataset_info(ds.id)
                    
                    results.append({
                        "dataset_id": ds.id,
                        "author": ds.author if hasattr(ds, 'author') else ds.id.split('/')[0],
                        "downloads": ds.downloads,
                        "likes": ds.likes,
                        "tags": ds.tags if hasattr(ds, 'tags') else [],
                        "size_gb": getattr(info, 'dataset_size', 0) / (1024**3),
                        "local": self.is_downloaded(ds.id)
                    })
                except Exception as e:
                    print(f"Error getting info for {ds.id}: {e}")
                    continue
                    
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def download_dataset(
        self,
        dataset_id: str,
        split: Optional[str] = None,
        subset: Optional[str] = None,
        token: Optional[str] = None,
        force: bool = False
    ) -> str:
        """Download dataset from HuggingFace Hub.
        
        Args:
            dataset_id: HuggingFace dataset ID
            split: Dataset split (train, test, validation)
            subset: Dataset subset/configuration
            token: HuggingFace API token
            force: Force re-download
            
        Returns:
            Local path to downloaded dataset
        """
        local_path = self.datasets_dir / dataset_id.replace("/", "--")
        
        if local_path.exists() and not force:
            print(f"Dataset {dataset_id} already exists at {local_path}")
            return str(local_path)
        
        print(f"Downloading {dataset_id} from HuggingFace Hub...")
        
        try:
            # Download dataset
            dataset = load_dataset(
                dataset_id,
                subset,
                split=split,
                token=token,
                cache_dir=str(self.datasets_dir / "cache")
            )
            
            # Save to disk
            local_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(local_path))
            
            # Save metadata
            metadata = {
                "dataset_id": dataset_id,
                "split": split,
                "subset": subset,
                "num_rows": len(dataset) if isinstance(dataset, Dataset) else sum(len(ds) for ds in dataset.values()),
                "columns": dataset.column_names if isinstance(dataset, Dataset) else list(dataset.values())[0].column_names,
                "local_path": str(local_path)
            }
            
            with open(local_path / "modelops_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Dataset downloaded to {local_path}")
            return str(local_path)
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            raise
    
    def upload_local_dataset(
        self,
        file_path: str,
        dataset_name: str,
        text_column: str = "text",
        label_column: Optional[str] = None
    ) -> str:
        """Upload local dataset file (CSV, JSON, Parquet).
        
        Args:
            file_path: Path to local file
            dataset_name: Name for the dataset
            text_column: Name of text column
            label_column: Name of label column (optional)
            
        Returns:
            Local path to processed dataset
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Processing local file: {file_path}")
        
        # Load file based on extension
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.json' or file_path.suffix == '.jsonl':
            df = pd.read_json(file_path, lines=(file_path.suffix == '.jsonl'))
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_pandas(df)
        
        # Save to local registry
        local_path = self.datasets_dir / dataset_name
        local_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(local_path))
        
        # Save metadata
        metadata = {
            "dataset_id": dataset_name,
            "source_file": str(file_path),
            "num_rows": len(dataset),
            "columns": dataset.column_names,
            "text_column": text_column,
            "label_column": label_column,
            "local_path": str(local_path)
        }
        
        with open(local_path / "modelops_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Dataset saved to {local_path}")
        return str(local_path)
    
    def list_local_datasets(self) -> List[DatasetInfo]:
        """List all downloaded datasets.
        
        Returns:
            List of DatasetInfo objects
        """
        datasets = []
        
        for dataset_dir in self.datasets_dir.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name == "cache":
                continue
            
            try:
                # Load dataset
                dataset = load_from_disk(str(dataset_dir))
                
                # Read metadata
                metadata_file = dataset_dir / "modelops_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    dataset_id = metadata.get("dataset_id", dataset_dir.name.replace("--", "/"))
                else:
                    dataset_id = dataset_dir.name.replace("--", "/")
                
                # Calculate size
                size_bytes = sum(
                    f.stat().st_size 
                    for f in dataset_dir.rglob("*") 
                    if f.is_file()
                )
                
                # Handle both Dataset and DatasetDict
                if isinstance(dataset, DatasetDict):
                    num_rows = sum(len(ds) for ds in dataset.values())
                    columns = list(dataset.values())[0].column_names if dataset else []
                    splits = list(dataset.keys())
                else:
                    num_rows = len(dataset)
                    columns = dataset.column_names
                    splits = ["train"]
                
                datasets.append(DatasetInfo(
                    dataset_id=dataset_id,
                    local_path=str(dataset_dir),
                    num_rows=num_rows,
                    num_columns=len(columns),
                    columns=columns,
                    size_mb=size_bytes / (1024**2),
                    splits=splits,
                    downloaded=True
                ))
            except Exception as e:
                print(f"Error loading dataset {dataset_dir}: {e}")
                continue
        
        return datasets
    
    def is_downloaded(self, dataset_id: str) -> bool:
        """Check if dataset is already downloaded.
        
        Args:
            dataset_id: HuggingFace dataset ID
            
        Returns:
            True if dataset exists locally
        """
        local_path = self.datasets_dir / dataset_id.replace("/", "--")
        return local_path.exists()
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a local dataset.
        
        Args:
            dataset_id: Dataset ID or name
            
        Returns:
            True if deleted successfully
        """
        local_path = self.datasets_dir / dataset_id.replace("/", "--")
        
        if not local_path.exists():
            return False
        
        import shutil
        shutil.rmtree(local_path)
        print(f"✓ Deleted dataset {dataset_id}")
        return True
    
    def get_recommended_datasets(self, task: str = "fine-tuning") -> List[Dict[str, Any]]:
        """Get recommended datasets for specific tasks.
        
        Args:
            task: Task type
            
        Returns:
            List of recommended datasets
        """
        recommendations = {
            "fine-tuning": [
                "timdettmers/openassistant-guanaco",
                "yahma/alpaca-cleaned",
                "databricks/databricks-dolly-15k",
                "tatsu-lab/alpaca",
            ],
            "instruction": [
                "HuggingFaceH4/ultrachat_200k",
                "OpenAssistant/oasst1",
            ],
            "coding": [
                "codeparrot/github-code",
                "bigcode/the-stack-dedup",
            ]
        }
        
        dataset_ids = recommendations.get(task, recommendations["fine-tuning"])
        
        return [
            {
                "dataset_id": did,
                "task": task,
                "local": self.is_downloaded(did)
            }
            for did in dataset_ids
        ]
    
    def prepare_for_training(
        self,
        dataset_path: str,
        text_column: str = "text",
        max_samples: Optional[int] = None,
        validation_split: float = 0.1
    ) -> Dict[str, str]:
        """Prepare dataset for training.
        
        Args:
            dataset_path: Path to dataset
            text_column: Name of text column
            max_samples: Maximum number of samples
            validation_split: Validation split ratio
            
        Returns:
            Paths to train and validation splits
        """
        dataset = load_from_disk(dataset_path)
        
        # Handle DatasetDict
        if isinstance(dataset, DatasetDict):
            if "train" in dataset:
                dataset = dataset["train"]
            else:
                dataset = list(dataset.values())[0]
        
        # Limit samples if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        # Create train/validation split
        split_dataset = dataset.train_test_split(test_size=validation_split)
        
        # Save splits
        output_dir = Path(dataset_path).parent / f"{Path(dataset_path).name}_prepared"
        output_dir.mkdir(exist_ok=True)
        
        split_dataset.save_to_disk(str(output_dir))
        
        return {
            "train": str(output_dir / "train"),
            "validation": str(output_dir / "test")
        }
