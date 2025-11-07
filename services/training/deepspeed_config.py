"""DeepSpeed ZeRO configuration."""


def get_deepspeed_config(stage: int = 2) -> dict:
    """Get DeepSpeed ZeRO configuration.
    
    Args:
        stage: ZeRO stage (0, 1, 2, or 3)
    
    Returns:
        DeepSpeed configuration dictionary
    """
    return {
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": False
        },
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            } if stage >= 2 else None,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            } if stage == 3 else None,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8 if stage == 3 else None,
            "stage3_param_persistence_threshold": 1e6 if stage == 3 else None
        },
        "gradient_accumulation_steps": 4,
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }
