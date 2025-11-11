"""Training Orchestrator - Connects workflows to actual training execution."""
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio
import json

from services.training.qlora_service import QLoRATrainingService
from services.model_registry import ModelRegistry
from services.dataset_registry import DatasetRegistry
from db.repository import JobRepository
from db.database import SessionLocal


class TrainingOrchestrator:
    """Orchestrates end-to-end training workflows."""
    
    def __init__(
        self,
        models_dir: str = "./models",
        datasets_dir: str = "./datasets",
        output_dir: str = "./training_output"
    ):
        """Initialize training orchestrator.
        
        Args:
            models_dir: Directory for models
            datasets_dir: Directory for datasets
            output_dir: Directory for training outputs
        """
        self.model_registry = ModelRegistry(models_dir)
        self.dataset_registry = DatasetRegistry(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def prepare_training_environment(
        self,
        job_id: str,
        base_model: str,
        dataset_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Prepare environment for training.
        
        Args:
            job_id: Job identifier
            base_model: Model ID to fine-tune
            dataset_id: Dataset ID for training
            config: Training configuration
            
        Returns:
            Paths to model and dataset
        """
        # Ensure model is downloaded
        if not self.model_registry.is_downloaded(base_model):
            print(f"Downloading model {base_model}...")
            model_path = self.model_registry.download_model(base_model)
        else:
            model_path = str(
                self.model_registry.models_dir / base_model.replace("/", "--")
            )
        
        # Ensure dataset is downloaded
        if not self.dataset_registry.is_downloaded(dataset_id):
            print(f"Downloading dataset {dataset_id}...")
            dataset_path = self.dataset_registry.download_dataset(dataset_id)
        else:
            dataset_path = str(
                self.dataset_registry.datasets_dir / dataset_id.replace("/", "--")
            )
        
        # Prepare dataset splits
        prepared = self.dataset_registry.prepare_for_training(
            dataset_path,
            text_column=config.get("text_column", "text"),
            max_samples=config.get("max_samples"),
            validation_split=config.get("validation_split", 0.1)
        )
        
        return {
            "model_path": model_path,
            "train_dataset": prepared["train"],
            "val_dataset": prepared["validation"],
            "output_dir": str(self.output_dir / job_id)
        }
    
    async def execute_qlora_training(
        self,
        job_id: str,
        base_model: str,
        dataset_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute QLoRA training workflow.
        
        Args:
            job_id: Job identifier
            base_model: Model ID to fine-tune
            dataset_id: Dataset ID for training
            config: Training configuration
            
        Returns:
            Training results
        """
        # Update job status
        db = SessionLocal()
        try:
            repo = JobRepository(db)
            await repo.update_job_status(job_id, "running")
            
            # Prepare environment
            paths = await self.prepare_training_environment(
                job_id, base_model, dataset_id, config
            )
            
            # Build training config
            training_config = {
                "base_model": paths["model_path"],
                "dataset_path": paths["train_dataset"],
                "output_dir": paths["output_dir"],
                "lora_rank": config.get("lora_rank", 8),
                "lora_alpha": config.get("lora_alpha", 16),
                "lora_dropout": config.get("lora_dropout", 0.1),
                "num_epochs": config.get("num_epochs", 3),
                "batch_size": config.get("batch_size", 2),
                "learning_rate": config.get("learning_rate", 2e-4),
                "max_seq_length": config.get("max_seq_length", 512),
                "text_column": config.get("text_column", "text"),
                "experiment_name": f"job_{job_id}",
                "target_modules": config.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj"]
                )
            }
            
            # Execute training
            service = QLoRATrainingService(training_config)
            results = service.run()
            
            # Save results
            results_file = Path(paths["output_dir"]) / "training_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Update job with results
            await repo.update_job_metrics(job_id, results)
            await repo.update_job_status(job_id, "completed")
            
            return results
            
        except Exception as e:
            # Update job with error
            if 'repo' in locals():
                await repo.update_job_status(job_id, "failed", error=str(e))
            raise
        finally:
            db.close()
    
    async def execute_quantization(
        self,
        job_id: str,
        model_path: str,
        method: str = "awq",
        bits: int = 4
    ) -> Dict[str, Any]:
        """Execute model quantization.
        
        Args:
            job_id: Job identifier
            model_path: Path to model to quantize
            method: Quantization method (awq, gptq, gguf)
            bits: Number of bits (4, 8)
            
        Returns:
            Quantization results
        """
        from services.quantization.quantization_service import QuantizationService
        
        output_path = str(self.output_dir / job_id / "quantized")
        
        service = QuantizationService()
        
        if method == "awq":
            result = await service.quantize_awq(model_path, output_path, bits)
        elif method == "gptq":
            result = await service.quantize_gptq(model_path, output_path, bits)
        elif method == "gguf":
            result = await service.export_gguf(model_path, output_path)
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        return result
    
    async def deploy_model(
        self,
        model_path: str,
        backend: str = "ollama",
        port: int = 11434
    ) -> Dict[str, Any]:
        """Deploy model for inference.
        
        Args:
            model_path: Path to model
            backend: Inference backend (ollama, tgi, vllm)
            port: Service port
            
        Returns:
            Deployment information
        """
        from services.inference.inference_manager import InferenceManager
        
        manager = InferenceManager()
        
        if backend == "ollama":
            deployment = await manager.deploy_ollama(model_path, port)
        elif backend == "tgi":
            deployment = await manager.deploy_tgi(model_path, port)
        elif backend == "vllm":
            deployment = await manager.deploy_vllm(model_path, port)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        return deployment
