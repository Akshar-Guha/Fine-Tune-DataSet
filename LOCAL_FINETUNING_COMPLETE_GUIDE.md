# 🚀 Complete Local LLM Fine-Tuning Guide

**Everything you need to fine-tune LLMs locally with full RLHF support!**

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Complete Workflow](#complete-workflow)
5. [Model & Dataset Management](#model--dataset-management)
6. [Fine-Tuning (QLoRA)](#fine-tuning-qlora)
7. [Quantization](#quantization)
8. [Model Deployment](#model-deployment)
9. [RLHF Training](#rlhf-training)
10. [Frontend Dashboard](#frontend-dashboard)
11. [Troubleshooting](#troubleshooting)

---

## 🎯 Quick Start

```bash
# 1. Navigate to modelops directory
cd "s:/projects/Fine Tunning/modelops"

# 2. Install dependencies
pip install poetry
poetry install

# 3. Start all services
python scripts/start_local_platform.py

# 4. Access the platform
# - API: http://localhost:8000/docs
# - Frontend: http://localhost:3000
# - MLflow: http://localhost:5000
```

---

## 💻 System Requirements

### Minimum Requirements
- **CPU**: 4 cores (8 recommended)
- **RAM**: 16GB (32GB recommended)
- **Storage**: 50GB free space
- **GPU**: Optional (NVIDIA with CUDA for faster training)

### Recommended for Best Experience
- **GPU**: NVIDIA RTX 3060 or better (12GB+ VRAM)
- **RAM**: 32GB+
- **Storage**: SSD with 100GB+ free

### Supported Operating Systems
- Windows 10/11
- Linux (Ubuntu 20.04+)
- macOS (M1/M2 with MPS support)

---

## 📦 Installation

### Step 1: Clone and Setup

```bash
cd "s:/projects/Fine Tunning/modelops"
pip install poetry
poetry install
```

### Step 2: Install Optional Components

```bash
# For GPU acceleration (NVIDIA)
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For quantization (AWQ)
poetry run pip install autoawq

# For quantization (GPTQ)
poetry run pip install auto-gptq

# For Ollama deployment
# Download from: https://ollama.ai/download

# For llama.cpp (CPU inference)
poetry run pip install llama-cpp-python
```

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Key settings:
# - MLFLOW_TRACKING_URI=http://localhost:5000
# - DATA_DIR=./data
# - MODEL_DIR=./models
```

---

## 🔄 Complete Workflow

### End-to-End Example: Fine-Tune TinyLlama

```python
from services.model_registry import ModelRegistry
from services.dataset_registry import DatasetRegistry
from services.training_orchestrator import TrainingOrchestrator

# 1. Initialize services
model_reg = ModelRegistry()
dataset_reg = DatasetRegistry()
orchestrator = TrainingOrchestrator()

# 2. Download model
print("📥 Downloading TinyLlama...")
model_path = model_reg.download_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 3. Download/Upload dataset
print("📥 Downloading dataset...")
dataset_path = dataset_reg.download_dataset("timdettmers/openassistant-guanaco")

# OR upload your own dataset
# dataset_path = dataset_reg.upload_local_dataset(
#     "my_data.csv",
#     "my_training_data",
#     text_column="text"
# )

# 4. Start fine-tuning
print("🏋️ Starting QLoRA fine-tuning...")
config = {
    "lora_rank": 8,
    "lora_alpha": 16,
    "num_epochs": 3,
    "batch_size": 2,
    "learning_rate": 2e-4,
    "max_seq_length": 512
}

results = await orchestrator.execute_qlora_training(
    job_id="job_001",
    base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset_id="timdettmers/openassistant-guanaco",
    config=config
)

print(f"✅ Training complete!")
print(f"📊 Perplexity: {results['perplexity']}")
print(f"💾 Model saved: {results['model_path']}")

# 5. Quantize model (optional)
print("⚙️ Quantizing model...")
from services.quantization.quantization_service import QuantizationService

quant_service = QuantizationService()
quant_result = await quant_service.export_gguf(
    results['model_path'],
    "./quantized/tinyllama_q4",
    quantization_type="q4_k_m"
)

print(f"✅ Quantization complete!")
print(f"💾 Size: {quant_result['size_mb']:.2f} MB")

# 6. Deploy for inference
print("🚀 Deploying model...")
from services.inference.inference_manager import InferenceManager

inference = InferenceManager()
deployment = await inference.deploy_ollama(
    quant_result['output_path'],
    model_name="my-tinyllama"
)

print(f"✅ Deployed!")
print(f"🌐 API: {deployment['api_endpoint']}")

# 7. Test inference
import requests

response = requests.post(
    deployment['api_endpoint'],
    json={
        "model": "my-tinyllama",
        "prompt": "Hello, how are you?",
        "stream": False
    }
)

print(f"🤖 Response: {response.json()['response']}")
```

---

## 📚 Model & Dataset Management

### Browse and Search Models

```python
from services.model_registry import ModelRegistry

registry = ModelRegistry()

# Search for small models
models = registry.search_models(
    query="llama",
    max_params=2,  # Max 2B parameters
    sort="downloads",
    limit=10
)

for model in models:
    print(f"📦 {model['model_id']}")
    print(f"   Params: {model['params_billions']:.1f}B")
    print(f"   Downloads: {model['downloads']:,}")
    print()

# Get recommended models
recommended = registry.get_recommended_models("fine-tuning")
for rec in recommended:
    print(f"⭐ {rec['model_id']}")
```

### Download Models

```python
# Download specific model
model_path = registry.download_model(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    force=False  # Skip if already downloaded
)

# List local models
local_models = registry.list_local_models()
for model in local_models:
    print(f"💾 {model.model_id}")
    print(f"   Size: {model.size_gb:.2f} GB")
    print(f"   Path: {model.local_path}")
```

### Manage Datasets

```python
from services.dataset_registry import DatasetRegistry

dataset_reg = DatasetRegistry()

# Search datasets
datasets = dataset_reg.search_datasets(
    query="instruction",
    task="text-generation",
    limit=10
)

# Download dataset
dataset_path = dataset_reg.download_dataset(
    "timdettmers/openassistant-guanaco",
    split="train"
)

# Upload your own data
dataset_path = dataset_reg.upload_local_dataset(
    file_path="./my_data.csv",
    dataset_name="my_custom_dataset",
    text_column="instruction",
    label_column="response"
)

# List local datasets
local_datasets = dataset_reg.list_local_datasets()
for ds in local_datasets:
    print(f"📊 {ds.dataset_id}")
    print(f"   Rows: {ds.num_rows:,}")
    print(f"   Columns: {ds.columns}")
```

---

## 🏋️ Fine-Tuning (QLoRA)

### Basic Fine-Tuning

```python
from services.training.qlora_service import QLoRATrainingService

config = {
    "base_model": "./models/TinyLlama--TinyLlama-1.1B-Chat-v1.0",
    "dataset_path": "./datasets/openassistant-guanaco",
    "output_dir": "./fine_tuned/my_model",
    
    # LoRA configuration
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    
    # Training hyperparameters
    "num_epochs": 3,
    "batch_size": 2,
    "grad_accum_steps": 8,
    "learning_rate": 2e-4,
    "max_seq_length": 512,
    
    # Dataset configuration
    "text_column": "text",
    
    # Experiment tracking
    "experiment_name": "my_first_finetune"
}

service = QLoRATrainingService(config)
results = service.run()

print(f"Training Loss: {results['train_loss']}")
print(f"Eval Loss: {results['eval_loss']}")
print(f"Perplexity: {results['perplexity']}")
```

### Advanced Configuration

```python
# For larger models (7B+)
advanced_config = {
    "base_model": "./models/mistral-7b",
    "dataset_path": "./datasets/my_data",
    "output_dir": "./fine_tuned/mistral_custom",
    
    # Aggressive memory optimization
    "batch_size": 1,
    "grad_accum_steps": 16,
    "max_seq_length": 256,
    
    # Higher rank for better performance
    "lora_rank": 16,
    "lora_alpha": 32,
    
    # More training
    "num_epochs": 5,
    "learning_rate": 1e-4,
    
    # Monitoring
    "logging_steps": 5,
    "save_steps": 50,
    "eval_steps": 50
}
```

---

## ⚙️ Quantization

### AWQ Quantization (GPU)

```python
from services.quantization.quantization_service import QuantizationService

service = QuantizationService()

result = await service.quantize_awq(
    model_path="./fine_tuned/my_model",
    output_path="./quantized/my_model_awq",
    bits=4,
    group_size=128
)

print(f"Quantized size: {result['size_mb']:.2f} MB")
```

### GPTQ Quantization (GPU)

```python
result = await service.quantize_gptq(
    model_path="./fine_tuned/my_model",
    output_path="./quantized/my_model_gptq",
    bits=4,
    group_size=128
)
```

### GGUF Export (CPU Inference)

```python
result = await service.export_gguf(
    model_path="./fine_tuned/my_model",
    output_path="./quantized/my_model_gguf",
    quantization_type="q4_k_m"  # Options: q4_k_m, q5_k_m, q8_0
)

# Use with llama.cpp or Ollama
```

---

## 🚀 Model Deployment

### Ollama Deployment (Recommended for Local)

```python
from services.inference.inference_manager import InferenceManager

manager = InferenceManager()

deployment = await manager.deploy_ollama(
    model_path="./quantized/my_model_gguf/model_q4_k_m.gguf",
    port=11434,
    model_name="my-custom-model"
)

# Test inference
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "my-custom-model",
        "prompt": "Explain quantum computing",
        "stream": False
    }
)

print(response.json()['response'])
```

### TGI Deployment (GPU, Production)

```python
deployment = await manager.deploy_tgi(
    model_path="./quantized/my_model_awq",
    port=8080,
    num_shard=1  # GPU parallelism
)

# Use OpenAI-compatible API
response = requests.post(
    "http://localhost:8080/generate",
    json={
        "inputs": "What is machine learning?",
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7
        }
    }
)
```

### vLLM Deployment (High Throughput)

```python
deployment = await manager.deploy_vllm(
    model_path="./fine_tuned/my_model",
    port=8000,
    tensor_parallel_size=1
)

# OpenAI-compatible API
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "my-custom-model",
        "prompt": "Hello, world!",
        "max_tokens": 128
    }
)
```

---

## 🎓 RLHF Training

### Basic RLHF Workflow

```python
from services.rlhf.rlhf_trainer import RLHFTrainer

rlhf_config = {
    "learning_rate": 1.41e-5,
    "batch_size": 4,
    "mini_batch_size": 1,
    "ppo_epochs": 4,
    "epochs": 1,
    "max_new_tokens": 128
}

trainer = RLHFTrainer(rlhf_config)

# Step 1: Train reward model
reward_model_path = trainer.train_reward_model(
    model_path="./fine_tuned/my_model",
    preference_dataset=preference_data,
    output_dir="./rlhf/reward_model"
)

# Step 2: Train with PPO
results = trainer.train_with_ppo(
    model_path="./fine_tuned/my_model",
    reward_model_path=reward_model_path,
    dataset=prompts_dataset,
    output_dir="./rlhf/aligned_model"
)

print(f"Final reward: {results['final_reward']}")
print(f"Model saved: {results['model_path']}")
```

---

## 🎨 Frontend Dashboard

### Start the Dashboard

```bash
# From modelops directory
cd frontend
npm install
npm run dev

# Access at http://localhost:3000
```

### Features

1. **Model Browser** - Search and download models
2. **Dataset Manager** - Upload and manage datasets
3. **Training Monitor** - Real-time training progress
4. **Model Comparison** - Compare multiple fine-tuned models
5. **Deployment Manager** - Deploy and test models
6. **Metrics Dashboard** - View training metrics and charts

---

## 🛠️ Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
config["batch_size"] = 1
config["grad_accum_steps"] = 16

# Reduce sequence length
config["max_seq_length"] = 256

# Use smaller LoRA rank
config["lora_rank"] = 4
config["lora_alpha"] = 8
```

### Slow Training

```bash
# Check GPU usage
nvidia-smi

# Enable gradient checkpointing (already enabled by default)
# Use mixed precision (already using fp16)

# Consider using smaller model
# TinyLlama (1.1B) instead of Llama-7B
```

### Model Download Fails

```python
# Use HuggingFace token for gated models
model_path = registry.download_model(
    "meta-llama/Llama-2-7b-hf",
    token="hf_YOUR_TOKEN_HERE"
)
```

### Quantization Errors

```bash
# Install specific versions
pip install autoawq==0.1.6
pip install auto-gptq==0.5.0

# For GGUF, ensure llama.cpp is installed
pip install llama-cpp-python
```

---

## 📊 Example: Complete Pipeline

```python
import asyncio
from services.model_registry import ModelRegistry
from services.dataset_registry import DatasetRegistry
from services.training_orchestrator import TrainingOrchestrator
from services.quantization.quantization_service import QuantizationService
from services.inference.inference_manager import InferenceManager

async def complete_pipeline():
    # Initialize
    model_reg = ModelRegistry()
    dataset_reg = DatasetRegistry()
    orchestrator = TrainingOrchestrator()
    quant_service = QuantizationService()
    inference = InferenceManager()
    
    # 1. Get model and dataset
    print("📥 Step 1: Downloading resources...")
    model_path = model_reg.download_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    dataset_path = dataset_reg.download_dataset("timdettmers/openassistant-guanaco")
    
    # 2. Fine-tune
    print("🏋️ Step 2: Fine-tuning...")
    config = {
        "lora_rank": 8,
        "num_epochs": 1,  # Quick demo
        "batch_size": 2,
        "learning_rate": 2e-4
    }
    
    results = await orchestrator.execute_qlora_training(
        "demo_job",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "timdettmers/openassistant-guanaco",
        config
    )
    
    # 3. Quantize
    print("⚙️ Step 3: Quantizing...")
    quant_result = await quant_service.export_gguf(
        results['model_path'],
        "./quantized/demo_model"
    )
    
    # 4. Deploy
    print("🚀 Step 4: Deploying...")
    deployment = await inference.deploy_ollama(
        quant_result['output_path'],
        model_name="demo-model"
    )
    
    print("✅ Pipeline complete!")
    print(f"🌐 API: {deployment['api_endpoint']}")
    print(f"🎯 Model: {deployment['model_name']}")

# Run pipeline
asyncio.run(complete_pipeline())
```

---

## 🎯 Next Steps

1. **Explore Models** - Try different base models (Phi-2, StableLM, Gemma)
2. **Custom Datasets** - Prepare your own instruction datasets
3. **Hyperparameter Tuning** - Experiment with LoRA ranks, learning rates
4. **RLHF** - Align models with human preferences
5. **Production** - Deploy models with TGI or vLLM for high throughput

---

## 📞 Support

- **Documentation**: See individual service files for detailed API docs
- **Issues**: Check the TROUBLESHOOTING section
- **Community**: Share your fine-tuned models!

**Happy Fine-Tuning! 🚀**
