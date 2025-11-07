# ModelOps Lightweight Edition - Quick Start
This guide will help you get the lightweight ModelOps platform running on your laptop in under 5 minutes.

## Prerequisites

- Python 3.10 or 3.11
- 8GB+ RAM (16GB recommended)
- 10GB+ free disk space
- (Optional) CUDA-capable GPU for training!
- Poetry (Python package manager)
- NVIDIA GPU (optional, for training)

## 🚀 Installation

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-org/modelops.git
cd modelops

# Install Python dependencies
pip install poetry
poetry install

# Activate virtual environment
poetry shell
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional for local dev)
nano .env
```

### 3. Start Infrastructure

```bash
# Start all services
docker-compose -f infra/docker/docker-compose.yml up -d

# Wait for services to be healthy (30-60 seconds)
docker-compose -f infra/docker/docker-compose.yml ps

# Initialize database
python scripts/init_db.py
```

## ✅ Verify Installation

Check all services are running:

```bash
# Check PostgreSQL
docker-compose -f infra/docker/docker-compose.yml exec postgres pg_isready

# Check MinIO
curl http://localhost:9000/minio/health/live

# Check Temporal
curl http://localhost:8088

# Check MLflow
curl http://localhost:5000
```

Access web interfaces:
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)
- **Temporal UI**: http://localhost:8088
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686

## 📊 Example Workflows

### 1. Add a Dataset

```bash
# Using CLI
modelops dataset add \
  --source ./data/training.parquet \
  --name my_dataset \
  --create-embeddings

# Using Python SDK
python -c "
from storage.delta_lake_client import DeltaLakeClient
import pandas as pd
import pyarrow as pa

# Load your data
df = pd.read_csv('data.csv')
table = pa.Table.from_pandas(df)

# Write to Delta Lake
client = DeltaLakeClient()
client.write_dataset('my_dataset', table)
"
```

### 2. Submit Training Job

```bash
# Create training config
cat > qlora_config.json << EOF
{
  "workflow": "qlora_training",
  "config": {
    "base_model": "meta-llama/Llama-2-7b-hf",
    "dataset_id": "my_dataset",
    "rank": 8,
    "alpha": 16,
    "epochs": 3,
    "auto_quantize": true
  }
}
EOF

# Submit job
modelops job submit training \
  --config qlora_config.json \
  --dataset my_dataset \
  --base-model meta-llama/Llama-2-7b-hf
```

### 3. Create RAG System

```bash
modelops rag create \
  --dataset my_dataset \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --chunk-size 512
```

### 4. Deploy Model

```bash
modelops deploy create \
  --artifact-id <your-artifact-id> \
  --backend tgi \
  --replicas 2
```

## 🧪 Test the Platform

### Storage Test

```python
from storage.minio_client import MinIOClient
from storage.delta_lake_client import DeltaLakeClient
from storage.lancedb_client import LanceDBClient

# Test MinIO
minio = MinIOClient()
minio.ensure_bucket("test-bucket")
print("✓ MinIO working")

# Test Delta Lake
delta = DeltaLakeClient()
import pandas as pd
import pyarrow as pa
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
delta.write_dataset("test", pa.Table.from_pandas(df))
print("✓ Delta Lake working")

# Test LanceDB
lance = LanceDBClient()
docs = [{"id": "1", "text": "Hello world"}]
lance.create_index("test", docs)
print("✓ LanceDB working")
```

### Artifact Registry Test

```python
from artifacts.registry.manager import ArtifactRegistry
from artifacts.schemas.base import *
from datetime import datetime

registry = ArtifactRegistry()

# Create test manifest
manifest = ArtifactManifest(
    artifact_id="test_001",
    name="Test Adapter",
    type=ArtifactType.ADAPTER,
    adapter=AdapterConfig(type="lora", rank=8, alpha=16, target_modules=["q_proj"]),
    base_model=BaseModelInfo(name="llama-2-7b", revision="main", sha256="abc123", size_bytes=1000000),
    dataset=DatasetInfo(id="test", delta_version="1", delta_snapshot_id=1, sha256="def456", num_rows=100, num_columns=3),
    training=TrainingInfo(
        workflow_id="wf_001", temporal_run_id="tr_001", mlflow_run_id="ml_001",
        duration_seconds=3600, gpu_type="A100", num_gpus=1, total_steps=1000, final_loss=0.5
    ),
    storage_uri="s3://test/adapter",
    size_bytes=100000,
    provenance=ProvenanceInfo(git_commit="abc123", created_by="test", trace_id="trace_001"),
    governance=GovernanceInfo(status=GovernanceStatus.DEV),
    signature=SignatureInfo(public_key="test", signature="test")
)

artifact_id = registry.register(manifest)
print(f"✓ Artifact registered: {artifact_id}")

# Retrieve artifact
retrieved = registry.get(artifact_id)
print(f"✓ Artifact retrieved: {retrieved.name}")
```

## 📝 Next Steps

1. **Explore Examples**: Check `notebooks/` for Jupyter examples
2. **Read Documentation**: See `docs/` for detailed guides
3. **Try Workflows**: Experiment with different ML workflows
4. **Join Community**: Contribute to the project

## 🐛 Troubleshooting

### Services won't start

```bash
# Check logs
docker-compose -f infra/docker/docker-compose.yml logs

# Restart specific service
docker-compose -f infra/docker/docker-compose.yml restart postgres

# Clean restart
docker-compose -f infra/docker/docker-compose.yml down -v
docker-compose -f infra/docker/docker-compose.yml up -d
```

### Database connection errors

```bash
# Verify PostgreSQL is running
docker-compose -f infra/docker/docker-compose.yml ps postgres

# Check connection
docker-compose -f infra/docker/docker-compose.yml exec postgres psql -U postgres -c "SELECT 1"

# Reinitialize
python scripts/init_db.py
```

### Storage issues

```bash
# Check MinIO
docker-compose -f infra/docker/docker-compose.yml exec minio mc admin info local

# Create buckets manually
docker-compose -f infra/docker/docker-compose.yml exec minio mc mb local/modelops
```

## 📚 Resources

- **Architecture**: See `docs/architecture.md`
- **API Reference**: See `docs/api_reference.md`
- **Workflows**: See `docs/workflows.md`
- **Deployment**: See `docs/deployment.md`

## 🤝 Getting Help

- GitHub Issues: https://github.com/your-org/modelops/issues
- Documentation: https://modelops.readthedocs.io
- Community: https://discord.gg/modelops

---

**Ready to build production ML systems? Let's go! 🚀**
