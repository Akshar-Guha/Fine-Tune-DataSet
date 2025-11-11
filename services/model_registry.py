"""Model Registry Service - Fetch and manage LLMs locally."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from huggingface_hub import snapshot_download, HfApi, model_info
from transformers import AutoConfig
import torch


@dataclass
class ModelInfo:
    """Model metadata."""
    model_id: str
    local_path: str
    model_type: str
    params_count: Optional[int]
    quantization: Optional[str]
    downloaded: bool
    size_gb: float


class ModelRegistry:
    """Manage local LLM models with HuggingFace Hub integration."""
    
    def __init__(self, models_dir: str = "./models"):
        """Initialize model registry.
        
        Args:
            models_dir: Local directory for storing models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.models_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.api = HfApi()
        
    def search_models(
        self,
        query: str = "",
        task: str = "text-generation",
        max_params: Optional[int] = None,
        sort: str = "downloads",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search HuggingFace Hub for models.
        
        Args:
            query: Search query
            task: Model task (text-generation, text-classification, etc.)
            max_params: Maximum number of parameters (in billions)
            sort: Sort by (downloads, likes, updated)
            limit: Maximum results
            
        Returns:
            List of model metadata
        """
        models = self.api.list_models(
            filter=task,
            search=query,
            sort=sort,
            limit=limit * 2  # Get more for filtering
        )
        
        results = []
        for model in models:
            if len(results) >= limit:
                break
                
            try:
                info = model_info(model.modelId)
                
                # Estimate parameters from config
                params = self._estimate_params(model.modelId)
                
                # Filter by max params
                if max_params and params:
                    if params > max_params * 1e9:
                        continue
                
                results.append({
                    "model_id": model.modelId,
                    "author": model.author if hasattr(model, 'author') else model.modelId.split('/')[0],
                    "downloads": model.downloads,
                    "likes": model.likes,
                    "params_billions": params / 1e9 if params else None,
                    "tags": model.tags,
                    "library": info.library_name if hasattr(info, 'library_name') else "unknown",
                    "local": self.is_downloaded(model.modelId)
                })
            except Exception as e:
                print(f"Error getting info for {model.modelId}: {e}")
                continue
                
        return results
    
    def _estimate_params(self, model_id: str) -> Optional[int]:
        """Estimate model parameter count from config.
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            Estimated parameter count or None
        """
        try:
            config = AutoConfig.from_pretrained(model_id)
            
            # Common parameter estimation formulas
            if hasattr(config, 'num_parameters'):
                return config.num_parameters
            
            # For transformer models
            if hasattr(config, 'hidden_size') and hasattr(config, 'num_hidden_layers'):
                hidden_size = config.hidden_size
                num_layers = config.num_hidden_layers
                vocab_size = getattr(config, 'vocab_size', 32000)
                
                # Rough estimation
                # Embedding: vocab_size * hidden_size
                # Each layer: ~4 * hidden_size^2 (attention + ffn)
                params = vocab_size * hidden_size
                params += num_layers * (4 * hidden_size * hidden_size)
                
                return params
                
        except Exception:
            pass
            
        return None
    
    def download_model(
        self,
        model_id: str,
        quantization: Optional[str] = None,
        token: Optional[str] = None,
        revision: str = "main",
        force: bool = False
    ) -> str:
        """Download model from HuggingFace Hub.
        
        Args:
            model_id: HuggingFace model ID (e.g., "TinyLlama/TinyLlama-1.1B")
            quantization: Quantization type (4bit, 8bit, None)
            token: HuggingFace API token
            revision: Model revision (branch/tag)
            force: Force re-download
            
        Returns:
            Local path to downloaded model
        """
        local_path = self.models_dir / model_id.replace("/", "--")
        
        if local_path.exists() and not force:
            print(f"Model {model_id} already exists at {local_path}")
            return str(local_path)
        
        print(f"Downloading {model_id} from HuggingFace Hub...")
        
        try:
            downloaded_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=str(self.cache_dir),
                local_dir=str(local_path),
                token=token,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"]  # Skip unnecessary files
            )
            
            # Save metadata
            metadata = {
                "model_id": model_id,
                "revision": revision,
                "quantization": quantization,
                "downloaded_at": str(torch.cuda.Event(enable_timing=True)),
                "local_path": str(local_path)
            }
            
            with open(local_path / "modelops_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Model downloaded to {local_path}")
            return str(local_path)
            
        except Exception as e:
            print(f"✗ Download failed: {e}")
            raise
    
    def list_local_models(self) -> List[ModelInfo]:
        """List all downloaded models.
        
        Returns:
            List of ModelInfo objects
        """
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == "cache":
                continue
            
            # Read metadata
            metadata_file = model_dir / "modelops_metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    
                model_id = metadata.get("model_id", model_dir.name.replace("--", "/"))
            else:
                model_id = model_dir.name.replace("--", "/")
            
            # Calculate size
            size_bytes = sum(
                f.stat().st_size 
                for f in model_dir.rglob("*") 
                if f.is_file()
            )
            
            models.append(ModelInfo(
                model_id=model_id,
                local_path=str(model_dir),
                model_type=self._get_model_type(model_dir),
                params_count=self._estimate_params(model_id),
                quantization=None,
                downloaded=True,
                size_gb=size_bytes / (1024**3)
            ))
        
        return models
    
    def _get_model_type(self, model_path: Path) -> str:
        """Get model architecture type.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Model type string
        """
        config_file = model_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                return config.get("model_type", "unknown")
        return "unknown"
    
    def is_downloaded(self, model_id: str) -> bool:
        """Check if model is already downloaded.
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            True if model exists locally
        """
        local_path = self.models_dir / model_id.replace("/", "--")
        return local_path.exists()
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a local model.
        
        Args:
            model_id: HuggingFace model ID
            
        Returns:
            True if deleted successfully
        """
        local_path = self.models_dir / model_id.replace("/", "--")
        
        if not local_path.exists():
            return False
        
        import shutil
        shutil.rmtree(local_path)
        print(f"✓ Deleted model {model_id}")
        return True
    
    def get_recommended_models(self, use_case: str = "fine-tuning") -> List[Dict[str, Any]]:
        """Get recommended models for specific use cases.
        
        Args:
            use_case: Use case (fine-tuning, inference, etc.)
            
        Returns:
            List of recommended models
        """
        recommendations = {
            "fine-tuning": [
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "microsoft/phi-2",
                "stabilityai/stablelm-2-1_6b",
                "google/gemma-2b",
            ],
            "inference": [
                "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            ],
            "embeddings": [
                "BAAI/bge-small-en-v1.5",
                "sentence-transformers/all-MiniLM-L6-v2",
            ]
        }
        
        model_ids = recommendations.get(use_case, recommendations["fine-tuning"])
        
        return [
            {
                "model_id": mid,
                "use_case": use_case,
                "local": self.is_downloaded(mid)
            }
            for mid in model_ids
        ]
