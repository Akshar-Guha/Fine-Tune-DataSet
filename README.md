# ModelOps – Lightweight Laptop Edition (100% Free & Open Source)

**Owner:** Boss  
**Purpose:** Lightweight ModelOps platform optimized for 1B models on laptops

## 🚀 Features

- **Laptop-Friendly** - Optimized for running 1B models on consumer hardware
- **Lightweight Stack** - Minimal dependencies, maximum efficiency
- **Complete MLOps** - Fine-tuning (QLoRA), Quantization (AWQ/GGUF), Inference
- **Embedded Storage** - DuckDB + LanceDB (no external servers needed)
- **Local Orchestration** - Prefect for workflow management
- **CPU Inference** - llama.cpp for efficient CPU-based inference
- **Experiment Tracking** - MLflow with local SQLite backend
- **Easy Setup** - Single command to get started

## 📦 Lightweight Tech Stack

### Fine-Tuning
- **QLoRA with bitsandbytes** - 4-bit quantized training for 1B models
- **PEFT (LoRA)** - Parameter-efficient fine-tuning
- **PyTorch 2.x** - Latest ML framework

### Inference
- **llama.cpp** - Fast CPU inference with GGUF models
- **Ollama** - Alternative local inference engine

### Storage
- **DuckDB** - Embedded SQL database for datasets
- **LanceDB** - Embedded vector database (no server)

### Orchestration & Tracking
- **Prefect** - Lightweight workflow orchestration
- **MLflow** - Experiment tracking with local SQLite

### Quantization
- **AutoAWQ** - AWQ quantization for GPU
- **GGUF Export** - CPU-optimized quantization

## 🏗️ Simple Architecture

```
FastAPI REST API
        │
        ├── Fine-Tuning (QLoRA + 4-bit)
        │   └── MLflow Tracking
        │
        ├── Storage (Embedded)
        │   ├── DuckDB (Datasets)
        │   └── LanceDB (Vectors)
        │
        ├── Inference
        │   ├── llama.cpp (GGUF)
        │   └── Ollama (Alternative)
        │
        └── Orchestration
            └── Prefect (Local)
```

## 🚀 Quick Start

### Local Development

```bash
# 1. Clone repository
git clone https://github.com/your-org/modelops.git
cd modelops

# 2. Install dependencies
pip install poetry
poetry install

# 3. Start MLflow (optional)
docker-compose up -d mlflow

# 4. Run API server
poetry run uvicorn api.rest.main:app --reload --port 8000

# 5. Access API
# http://localhost:8000/docs
```

### Using CLI

```bash
# Add dataset
modelops dataset add \
  --source ./data/training.parquet \
  --name my_dataset \
  --create-embeddings

# Submit training job
modelops job submit training \
  --config configs/qlora.json \
  --dataset my_dataset \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Monitor job
modelops job status <job-id>

# Deploy model
modelops deploy create \
  --artifact-id <artifact-id> \
  --backend tgi \
  --replicas 2
```

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [Workflows Guide](docs/workflows.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment.md)
- [Plugin Development](docs/plugins.md)

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:

```bash
# MLflow Tracking
MLFLOW_TRACKING_URI=http://localhost:5000

# Prefect
PREFECT_API_URL=http://localhost:4200/api

# Data Directories
DATA_DIR=./data
MODEL_DIR=./models
DATASET_DIR=./datasets
```

## 🧪 Testing

```bash
# Run unit tests
poetry run pytest tests/unit

# Run integration tests
poetry run pytest tests/integration

# Run end-to-end tests
poetry run pytest tests/e2e
```

## 📊 Monitoring

Access services:
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Metrics**: http://localhost:8000/metrics

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

Built with amazing open-source tools:
- PyTorch, Hugging Face, llama.cpp, Prefect, DuckDB, LanceDB, MLflow
