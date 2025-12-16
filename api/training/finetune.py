"""Local fine-tuning implementation using QLoRA."""
import json
from typing import Dict, Any
from pathlib import Path

from sqlalchemy.orm import Session
from modelops.db.repository import JobRepository

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset


async def run_finetuning_job(job_id: str, job_config: Dict[str, Any], db: Session):
    """Run a fine-tuning job asynchronously."""
    try:
        # Update job status to running
        repo = JobRepository(db)
        await repo.update_job_status(job_id, "running")

        # Run the training
        result = await _run_training(job_config)

        # Update job with results
        await repo.complete_job(
            job_id,
            metrics=result.get("metrics"),
            artifacts=result.get("artifacts")
        )

    except Exception as e:
        # Update job with error
        repo = JobRepository(db)
        await repo.fail_job(job_id, str(e))


async def _run_training(job_config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the QLoRA fine-tuning process."""
    config = job_config.get("config", {})
    base_model = job_config["base_model"]
    dataset_id = job_config["dataset_id"]

    # Create output directory
    output_dir = Path("./artifacts") / f"finetune_{job_config['name'].replace(' ', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, config.get("bnb_4bit_compute_dtype", "float16")),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=config.get("lora_dropout", 0.1),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Load dataset
    if dataset_id.startswith("hf:"):
        dataset_name = dataset_id[3:]  # Remove hf: prefix
        dataset = load_dataset(dataset_name, split="train")
    else:
        # Assume local dataset file
        dataset_path = Path("./datasets") / dataset_id
        if dataset_path.suffix == ".json":
            dataset = load_dataset("json", data_files=str(dataset_path), split="train")
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

    # Prepare dataset for training
    def tokenize_function(examples):
        return tokenizer(
            examples[config.get("text_column", "text")],
            truncation=True,
            padding="max_length",
            max_length=config.get("max_seq_length", 512),
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Split into train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 2e-4),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 50),
        evaluation_strategy=config.get("evaluation_strategy", "steps"),
        eval_steps=config.get("eval_steps", 50),
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,
        report_to="none",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field=config.get("text_column", "text"),
        max_seq_length=config.get("max_seq_length", 512),
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model(str(output_dir))

    # Get training metrics
    training_logs = trainer.state.log_history
    final_metrics = {}
    if training_logs:
        final_log = training_logs[-1]
        for key, value in final_log.items():
            if isinstance(value, (int, float)):
                final_metrics[key] = value

    # Create artifacts info
    artifacts = {
        "model_path": str(output_dir),
        "tokenizer_path": str(output_dir),
        "config": config,
        "base_model": base_model,
        "dataset": dataset_id,
    }

    return {
        "metrics": final_metrics,
        "artifacts": artifacts,
    }
