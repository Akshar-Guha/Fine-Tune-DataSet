"""Lightweight QLoRA training service for 1B models."""
import os
from typing import Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import mlflow


class QLoRATrainingService:
    """Lightweight QLoRA training service optimized for 1B models on laptops."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize QLoRA training service.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model_and_tokenizer(self) -> None:
        """Load base model and tokenizer with 4-bit quantization."""
        # BitsAndBytes 4-bit config for efficient training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # Load model (no flash attention for simplicity)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.config.get("trust_remote_code", False),
            low_cpu_mem_usage=True
        )

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = False

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["base_model"],
            trust_remote_code=self.config.get("trust_remote_code", False)
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def apply_lora(self) -> None:
        """Apply LoRA configuration to the model."""
        lora_config = LoraConfig(
            r=self.config.get("lora_rank", 8),
            lora_alpha=self.config.get("lora_alpha", 16),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"]
            )
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_datasets(self) -> Dict[str, Any]:
        """Prepare training and evaluation datasets from DuckDB or local files.
        
        Returns:
            Dictionary with train and eval datasets
        """
        dataset_path = self.config.get("dataset_path")
        
        if dataset_path and os.path.exists(dataset_path):
            # Load from local file (parquet, csv, json)
            dataset = load_dataset(
                "parquet" if dataset_path.endswith(".parquet") else "json",
                data_files=dataset_path,
                split="train"
            )
        else:
            # Fallback to HuggingFace dataset
            dataset = load_dataset(
                self.config.get("dataset_name", "timdettmers/openassistant-guanaco"),
                split="train"
            )

        # Tokenize
        def tokenize_function(examples):
            text_key = self.config.get("text_column", "text")
            return self.tokenizer(
                examples[text_key],
                truncation=True,
                max_length=self.config.get("max_seq_length", 512),  # Reduced for 1B models
                padding="max_length"
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Split train/eval
        split = tokenized.train_test_split(test_size=0.1)
        return {
            "train": split["train"],
            "eval": split["test"]
        }

    def setup_trainer(self, datasets: Dict[str, Any]) -> None:
        """Setup HuggingFace Trainer (lightweight, no DeepSpeed).
        
        Args:
            datasets: Training and evaluation datasets
        """
        # Setup MLflow tracking
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(self.config.get("experiment_name", "qlora-training"))
        
        # Training arguments optimized for laptops
        training_args = TrainingArguments(
            output_dir=self.config.get("output_dir", "./results"),
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 2),  # Reduced for laptops
            gradient_accumulation_steps=self.config.get("grad_accum_steps", 8),  # Increased to compensate
            learning_rate=self.config.get("learning_rate", 2e-4),
            logging_steps=self.config.get("logging_steps", 10),
            save_steps=self.config.get("save_steps", 100),
            eval_steps=self.config.get("eval_steps", 100),
            warmup_ratio=self.config.get("warmup_ratio", 0.03),
            weight_decay=self.config.get("weight_decay", 0.01),
            fp16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",  # More memory efficient
            logging_dir="./logs",
            save_total_limit=2,  # Save space
            load_best_model_at_end=True,
            report_to=["mlflow"],
            max_grad_norm=0.3,  # Prevent gradient explosion
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["eval"],
            tokenizer=self.tokenizer
        )

    def train(self) -> Dict[str, Any]:
        """Execute training.
        
        Returns:
            Training results and metrics
        """
        # Train
        train_result = self.trainer.train()

        # Save model
        self.trainer.save_model()

        # Evaluate
        eval_results = self.trainer.evaluate()

        return {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_results["eval_loss"],
            "perplexity": torch.exp(torch.tensor(eval_results["eval_loss"])).item(),
            "model_path": self.config.get("output_dir", "./results")
        }

    def run(self) -> Dict[str, Any]:
        """Run complete training pipeline.
        
        Returns:
            Training results
        """
        with mlflow.start_run():
            # Log config
            mlflow.log_params({
                "base_model": self.config["base_model"],
                "lora_rank": self.config.get("lora_rank", 8),
                "batch_size": self.config.get("batch_size", 2),
                "learning_rate": self.config.get("learning_rate", 2e-4),
            })
            
            print("Loading model and tokenizer...")
            self.load_model_and_tokenizer()

            print("Applying LoRA...")
            self.apply_lora()

            print("Preparing datasets...")
            datasets = self.prepare_datasets()

            print("Setting up trainer...")
            self.setup_trainer(datasets)

            print("Starting training...")
            results = self.train()

            # Log results
            mlflow.log_metrics({
                "final_train_loss": results["train_loss"],
                "final_eval_loss": results["eval_loss"],
                "perplexity": results["perplexity"]
            })
            
            print("Training completed!")
            return results
