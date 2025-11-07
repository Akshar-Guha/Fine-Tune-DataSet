"""QLoRA training configuration."""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    """LoRA configuration."""
    rank: int = Field(8, ge=1, le=256)
    alpha: int = Field(16, ge=1)
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    target_modules: List[str] = ["q_proj", "v_proj"]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainingArgs(BaseModel):
    """Training arguments."""
    num_epochs: int = Field(3, ge=1)
    batch_size: int = Field(4, ge=1)
    gradient_accumulation_steps: int = Field(4, ge=1)
    learning_rate: float = Field(2e-4, gt=0.0)
    warmup_ratio: float = Field(0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(0.01, ge=0.0)
    max_grad_norm: float = Field(1.0, gt=0.0)
    save_steps: int = Field(100, ge=1)
    logging_steps: int = Field(10, ge=1)
    eval_steps: int = Field(100, ge=1)
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"


class DeepSpeedConfig(BaseModel):
    """DeepSpeed ZeRO configuration."""
    stage: int = Field(2, ge=0, le=3)
    offload_optimizer: bool = True
    offload_param: bool = False
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: int = 5e8
    stage3_prefetch_bucket_size: int = 5e8
    stage3_param_persistence_threshold: int = 1e6


class QLoRAConfig(BaseModel):
    """Complete QLoRA training configuration."""
    # Model config
    base_model: str
    model_revision: str = "main"
    trust_remote_code: bool = False

    # Dataset config
    dataset_id: str
    eval_dataset: Optional[str] = None
    max_seq_length: int = Field(2048, ge=128)

    # LoRA config
    lora_config: LoRAConfig = Field(default_factory=LoRAConfig)

    # Training config
    training_args: TrainingArgs = Field(default_factory=TrainingArgs)

    # DeepSpeed config
    use_deepspeed: bool = True
    deepspeed_config: Optional[DeepSpeedConfig] = Field(
        default_factory=DeepSpeedConfig
    )

    # Flash Attention
    use_flash_attention: bool = True

    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Post-training
    auto_quantize: bool = True
    quantization_method: str = "awq"

    # MLflow tracking
    experiment_name: str = "qlora_experiments"
    run_name: Optional[str] = None

    # Misc
    seed: int = 42
    num_workers: int = 4
