#!/usr/bin/env python3
"""
Test script for local GPT-2 fine-tuning using QLoRA.
This script demonstrates the complete fine-tuning workflow.
"""
import os
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset


def main():
    """Run GPT-2 fine-tuning test."""
    print("🚀 Starting GPT-2 Fine-Tuning Test")
    print("=" * 50)

    # Configuration
    config = {
        "base_model": "openai-community/gpt2",
        "dataset_name": "timdettmers/openassistant-guanaco",  # Small dataset for testing
        "text_column": "text",
        "output_dir": "./test_finetuned_gpt2",
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "num_epochs": 1,  # Just 1 epoch for testing
        "batch_size": 1,  # Small batch size for laptop
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "max_seq_length": 256,  # Smaller for testing
        "logging_steps": 5,
        "save_steps": 10,
        "eval_strategy": "steps",  # Updated parameter name
        "eval_steps": 10,
        "load_in_4bit": False,  # Disable 4-bit for CPU testing
        "use_lora": True,  # Use LoRA even without quantization
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load tokenizer
        print("🔤 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"],
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        print("🤖 Loading GPT-2 model...")
        if config["load_in_4bit"]:
            # Configure 4-bit quantization
            print("📦 Configuring 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=config["load_in_4bit"],
                bnb_4bit_compute_dtype=getattr(torch, config["bnb_4bit_compute_dtype"]),
                bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                config["base_model"],
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            # Prepare model for training with quantization
            model = prepare_model_for_kbit_training(model)
        else:
            # Load model without quantization for CPU training
            model = AutoModelForCausalLM.from_pretrained(
                config["base_model"],
                trust_remote_code=True,
            )
            # Use float32 for CPU training
            model = model.to(torch.float32)

        print(f"✅ Model loaded: {config['base_model']}")
        print(f"📊 Model parameters: {model.num_parameters()/1e6:.1f}M")
        print(f"🖥️ Device: {model.device}")
        print()

        # Apply LoRA if enabled
        if config["use_lora"]:
            print("🔧 Applying LoRA configuration...")
            # Configure LoRA
            lora_config = LoraConfig(
                r=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                target_modules=["c_attn"],  # GPT-2 attention modules
                lora_dropout=config["lora_dropout"],
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Apply LoRA
            model = get_peft_model(model, lora_config)
            print("✅ LoRA applied")
            print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")
        else:
            print("⚠️ Running full fine-tuning (no LoRA)")
        print()

        # Load dataset
        print("📊 Loading dataset...")
        dataset = load_dataset(config["dataset_name"], split="train[:100]")  # Small subset for testing
        print(f"✅ Loaded {len(dataset)} examples from {config['dataset_name']}")
        print(f"Sample: {dataset[0][config['text_column']][:100]}...")
        print()

        # Prepare dataset for training
        def tokenize_function(examples):
            return tokenizer(
                examples[config["text_column"]],
                truncation=True,
                padding="max_length",
                max_length=config["max_seq_length"],
            )

        print("🎯 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Split into train/eval
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Eval samples: {len(eval_dataset)}")
        print()

        # Training arguments
        print("⚙️ Configuring training arguments...")
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            logging_steps=config["logging_steps"],
            save_steps=config["save_steps"],
            eval_strategy=config["eval_strategy"],  # Updated parameter name
            eval_steps=config["eval_steps"],
            save_total_limit=2,
            load_best_model_at_end=True,
            fp16=False,  # Disable fp16 for CPU
            report_to="none",
        )

        # Initialize trainer
        print("🎓 Initializing trainer...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,  # Updated parameter name
        )

        # Start training
        print("🚀 Starting training...")
        trainer.train()

        # Save the model
        print("💾 Saving model...")
        trainer.save_model(str(output_dir))

        # Show training results
        training_results = trainer.state.log_history
        if training_results:
            print("\n📊 Final training metrics:")
            final_log = training_results[-1]
            for key, value in final_log.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")

        print("\n✅ Training complete!")
        print(f"📁 Model saved to: {output_dir}")

        # Test the model
        print("\n🧪 Testing fine-tuned model...")
        from transformers import pipeline

        pipe = pipeline(
            "text-generation",
            model=str(output_dir),
            tokenizer=tokenizer,
            device_map="auto",
            max_new_tokens=50,
        )

        test_prompts = [
            "Explain machine learning in simple terms:",
            "What is artificial intelligence?",
            "Write a short poem about coding:",
        ]

        print("🤖 Model outputs:")
        for prompt in test_prompts:
            print(f"\n📝 Prompt: {prompt}")
            output = pipe(prompt)[0]["generated_text"]
            if output.startswith(prompt):
                output = output[len(prompt):].strip()
            print(f"🤖 Response: {output[:100]}...")

        print("\n🎉 Fine-tuning test completed successfully!")
        print("📋 Next steps:")
        print("1. Your fine-tuned model is saved in:", output_dir)
        print("2. You can use this model for inference or deploy it")
        print("3. For Google Colab, the same code works with GPU acceleration")

    except Exception as e:
        print(f"❌ Error during fine-tuning: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
