"""Colab integration endpoints for GPU-accelerated fine-tuning."""
from typing import Dict, Any, Optional
import json
import math
import os
import shutil
import tempfile

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

router = APIRouter()


class ColabConfig(BaseModel):
    """Configuration for Colab fine-tuning."""
    base_model: str = Field(..., description="Base model to fine-tune")
    dataset_id: str = Field(..., description="Dataset ID to use")
    experiment_name: str = Field(..., description="Experiment name")
    lora_rank: int = Field(8, description="LoRA rank")
    lora_alpha: int = Field(16, description="LoRA alpha")
    num_epochs: int = Field(3, description="Number of training epochs")
    batch_size: int = Field(2, description="Batch size")
    learning_rate: float = Field(2e-4, description="Learning rate")
    max_seq_length: int = Field(512, description="Maximum sequence length")


class ColabJob(BaseModel):
    """Colab fine-tuning job."""
    job_id: str
    config: ColabConfig
    status: str = "created"
    colab_url: Optional[str] = None
    download_url: Optional[str] = None


LOCAL_HARDWARE_PROFILE = {
    "cpu": "AMD Ryzen 5 4600H (6C/12T)",
    "gpu": "NVIDIA GTX 1650",
    "gpu_vram_gb": 4,
    "ram_gb": 16,
    "os": "Windows 11",
    "notes": [
        "Discrete GPU with 4GB VRAM is memory constrained for modern LLM fine-tuning.",
        "Recommended to use Google Colab (T4/A100) for models above ~1B parameters or batch sizes >1.",
    ],
}


def _compute_gradient_accumulation(batch_size: int) -> int:
    """Derive gradient accumulation steps for stable training."""
    target_tokens = 32  # Aim for ~32 effective micro-batch updates
    grad_accum = max(1, math.ceil(target_tokens / max(batch_size, 1)))
    return min(grad_accum, 32)


def _estimate_training_outcomes(config: ColabConfig) -> Dict[str, Any]:
    """Estimate training outcomes for visualization and guidance."""
    effective_grad_steps = _compute_gradient_accumulation(config.batch_size)
    tokens_per_step = config.batch_size * config.max_seq_length * effective_grad_steps
    # Heuristic for total tokens processed (assuming ~5k samples per epoch)
    total_tokens = tokens_per_step * config.num_epochs * 5000 / max(config.max_seq_length, 1)

    runtime_minutes = max(10, round(total_tokens / 8.4e6, 1))
    gpu_memory_gb = round(4.2 + (config.batch_size * 0.35), 1)

    baseline = {
        "train_loss": 2.8,
        "eval_loss": 2.95,
        "perplexity": 19.1,
        "accuracy": 0.58,
    }

    improvement_factor = min(0.35, 0.08 * config.num_epochs + 0.01 * config.lora_rank)
    lr_adjustment = max(0.9, min(1.1, config.learning_rate / 2e-4))
    projected = {
        "train_loss": round(baseline["train_loss"] * (1 - improvement_factor * 0.9 * lr_adjustment), 3),
        "eval_loss": round(baseline["eval_loss"] * (1 - improvement_factor * 0.8 * lr_adjustment), 3),
        "perplexity": round(max(5.0, baseline["perplexity"] * (1 - improvement_factor * lr_adjustment)), 2),
        "accuracy": round(min(0.95, baseline["accuracy"] + improvement_factor * 0.25), 3),
    }

    chart_data = [
        {"metric": "Train Loss", "baseline": baseline["train_loss"], "projected": projected["train_loss"]},
        {"metric": "Eval Loss", "baseline": baseline["eval_loss"], "projected": projected["eval_loss"]},
        {"metric": "Perplexity", "baseline": baseline["perplexity"], "projected": projected["perplexity"]},
        {
            "metric": "Accuracy",
            "baseline": round(baseline["accuracy"] * 100, 1),
            "projected": round(projected["accuracy"] * 100, 1),
        },
    ]

    return {
        "estimates": {
            "runtime_minutes": runtime_minutes,
            "gpu_memory_gb": gpu_memory_gb,
            "effective_batch_tokens": tokens_per_step,
            "gradient_accumulation_steps": effective_grad_steps,
        },
        "baseline_metrics": baseline,
        "projected_metrics": projected,
        "chart": chart_data,
        "notes": [
            "Estimates assume Google Colab T4 GPU with 16GB VRAM.",
            "Perplexity/accuracy are heuristic projections to guide configuration tuning.",
        ],
    }


def _derive_runtime_recommendation(estimates: Dict[str, Any]) -> Dict[str, Any]:
    """Recommend local vs Colab runtime based on hardware profile and estimates."""
    required_gpu = estimates["gpu_memory_gb"]
    available_gpu = LOCAL_HARDWARE_PROFILE["gpu_vram_gb"]

    local_supported = required_gpu <= available_gpu * 0.9  # leave headroom for OS/driver usage

    if local_supported:
        recommendation = (
            "✅ Your laptop can attempt local fine-tuning, but expect longer runtimes. "
            "Keep batch size low and close other GPU-intensive apps."
        )
        preferred_runtime = "local"
    else:
        recommendation = (
            "⚠️ Estimated GPU memory requirements exceed your laptop's 4GB VRAM. "
            "Use Google Colab with a T4/A100 GPU for a smoother experience."
        )
        preferred_runtime = "colab"

    return {
        "local_supported": local_supported,
        "preferred_runtime": preferred_runtime,
        "recommendation": recommendation,
        "hardware_profile": LOCAL_HARDWARE_PROFILE,
    }


@router.post("/generate-notebook", response_model=Dict[str, Any])
async def generate_colab_notebook(config: ColabConfig):
    """Generate a Colab notebook for fine-tuning with the given
    configuration."""
    try:
        # Load the base notebook template
        template_path = os.path.join(
            os.path.dirname(__file__),
            "../../colab_finetuning.ipynb"
        )
        if not os.path.exists(template_path):
            raise HTTPException(status_code=404, detail="Colab notebook template not found")

        with open(template_path, 'r') as f:
            notebook = json.load(f)

        # Update configuration in the notebook
        config_cell = None
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code' and 'config = {' in ''.join(
                cell['source']
            ):
                config_cell = cell
                break

        if config_cell:
            grad_accum = _compute_gradient_accumulation(config.batch_size)
            # Update the config dictionary in the code
            config_code = f'''# Fine-tuning configuration
config = {{
    # Model settings
    "base_model": "{config.base_model}",
    "output_dir": "./fine_tuned_model",

    # Dataset settings
    "dataset_name": "{config.dataset_id}",
    "text_column": "text",

    # LoRA settings
    "lora_rank": {config.lora_rank},
    "lora_alpha": {config.lora_alpha},
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],

    # Training settings
    "num_epochs": {config.num_epochs},
    "batch_size": {config.batch_size},
    "gradient_accumulation_steps": {grad_accum},
    "learning_rate": {config.learning_rate},
    "max_seq_length": {config.max_seq_length},
    "logging_steps": 10,
    "save_steps": 50,
    "evaluation_strategy": "steps",
    "eval_steps": 50,

    # Memory optimization
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",

    # Experiment tracking
    "experiment_name": "{config.experiment_name}"
}}

print("Configuration loaded:")
for key, value in config.items():
    print(f"  {{key}}: {{value}}")'''

            config_cell['source'] = config_code.split('\n')

        # Create a temporary file for the customized notebook
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(notebook, f, indent=2)
            temp_path = f.name

        outcome_estimates = _estimate_training_outcomes(config)

        runtime_recommendation = _derive_runtime_recommendation(outcome_estimates["estimates"])

        return {
            "message": "Colab notebook generated successfully",
            "download_url": f"/api/v1/colab/download/{os.path.basename(temp_path)}",
            "instructions": [
                "1. Download the notebook using the URL above",
                "2. Open Google Colab (colab.research.google.com)",
                "3. Upload the notebook and connect to GPU runtime",
                "4. Run all cells in order",
                "5. Download your fine-tuned model at the end"
            ],
            "estimates": outcome_estimates["estimates"],
            "baseline_metrics": outcome_estimates["baseline_metrics"],
            "projected_metrics": outcome_estimates["projected_metrics"],
            "chart": outcome_estimates["chart"],
            "notes": outcome_estimates["notes"],
            **runtime_recommendation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Notebook generation failed: {str(e)}")


@router.get("/download/{filename}")
async def download_notebook(filename: str):
    """Download a generated Colab notebook."""
    try:
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        if not os.path.exists(temp_path):
            raise HTTPException(status_code=404, detail="Notebook not found")

        return FileResponse(
            temp_path,
            media_type='application/x-ipynb+json',
            filename=f"modelops_colab_finetuning.ipynb"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.post("/import-model")
async def import_colab_model(background_tasks: BackgroundTasks, model_zip_path: str):
    """Import a fine-tuned model from Colab."""
    try:
        if not os.path.exists(model_zip_path):
            raise HTTPException(status_code=404, detail="Model file not found")

        # Extract model to artifacts directory
        artifacts_dir = os.path.join(os.getcwd(), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        model_name = f"colab_import_{os.path.splitext(os.path.basename(model_zip_path))[0]}"
        extract_path = os.path.join(artifacts_dir, model_name)

        # Extract the zip file
        shutil.unpack_archive(model_zip_path, extract_path)

        # Register the model artifact
        background_tasks.add_task(register_imported_model, extract_path, model_name)

        return {
            "message": "Model import started",
            "model_path": extract_path,
            "status": "processing"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


async def register_imported_model(model_path: str, model_name: str):
    """Register an imported model in the artifact registry."""
    try:
        # This would integrate with your existing artifact management
        # For now, just log the successful import
        print(f"Model imported successfully: {model_name} at {model_path}")

        # TODO: Integrate with your artifacts.py to register the model

    except Exception as e:
        print(f"Failed to register imported model: {str(e)}")


@router.get("/templates")
async def get_colab_templates():
    """Get available Colab notebook templates."""
    return {
        "templates": [
            {
                "name": "QLoRA Fine-Tuning",
                "description": "GPU-accelerated fine-tuning with QLoRA",
                "file": "colab_finetuning.ipynb",
                "features": [
                    "4-bit quantization",
                    "LoRA adaptation",
                    "Automatic model download",
                    "GPU optimization"
                ]
            }
        ]
    }


@router.post("/validate-config")
async def validate_colab_config(config: ColabConfig):
    """Validate a Colab configuration before generation."""
    try:
        # Basic validation
        issues = []

        if not config.base_model:
            issues.append("base_model is required")

        if not config.dataset_id:
            issues.append("dataset_id is required")

        if config.lora_rank <= 0:
            issues.append("lora_rank must be positive")

        if config.num_epochs <= 0:
            issues.append("num_epochs must be positive")

        if config.learning_rate <= 0:
            issues.append("learning_rate must be positive")

        if issues:
            return {
                "valid": False,
                "issues": issues
            }

        outcome_estimates = _estimate_training_outcomes(config)

        runtime_recommendation = _derive_runtime_recommendation(outcome_estimates["estimates"])

        return {
            "valid": True,
            "estimated_time": f"~{outcome_estimates['estimates']['runtime_minutes']} minutes on T4 GPU",
            "estimated_cost": "Free (Google Colab)",
            "estimates": outcome_estimates["estimates"],
            "baseline_metrics": outcome_estimates["baseline_metrics"],
            "projected_metrics": outcome_estimates["projected_metrics"],
            "chart": outcome_estimates["chart"],
            "notes": outcome_estimates["notes"],
            **runtime_recommendation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")
