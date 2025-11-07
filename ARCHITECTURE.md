# ModelOps Architecture

Comprehensive architecture guide for the ModelOps platform.

## 🏗️ System Overview

ModelOps is a production-grade MLOps platform for LLM fine-tuning, quantization, and deployment built entirely on free, open-source tools.

### Design Principles

1. **100% Free Forever** - No enterprise editions or paid tiers
2. **Production-Grade** - Built for real workloads
3. **ACID Compliance** - All data operations are transactional
4. **Full Lineage** - Track every artifact from data to deployment
5. **Composable** - Mix and match components
6. **Observable** - Complete visibility into all operations

## 📐 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Gateway (Traefik)                │
│                  Authentication & Rate Limiting             │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┴─────────┐
    │                  │
┌───▼────────┐    ┌────▼──────────┐
│ Temporal   │    │ Argo          │
│ Workflows  │    │ Workflows     │
│ (Python)   │    │ (K8s Native)  │
└─────┬──────┘    └────┬──────────┘
      │                │
      └────────┬───────┘
               │
        ┌──────▼──────────────────────────────┐
        │          Core Services              │
        │  ┌─────────────┬─────────────┐     │
        │  │  Training   │  Inference  │     │
        │  │  DeepSpeed  │  TGI/vLLM   │     │
        │  └─────────────┴─────────────┘     │
        └──────┬──────────────────────────────┘
               │
    ┌──────────▼──────────────────────────────┐
    │         Data & Storage Layer            │
    │  ┌──────────────────────────────────┐  │
    │  │  MinIO  │ Delta  │ Lance │ Pg  │  │
    │  │  (S3)   │ Lake   │ DB    │ SQL │  │
    │  └──────────────────────────────────┘  │
    └─────────────────────────────────────────┘
               │
        ┌──────▼───────┐
        │ Observability│
        │ OTel + Prom  │
        │ + Grafana    │
        └──────────────┘
```

## 🗄️ Data Layer Architecture

### Storage Hierarchy

```
MinIO (Object Storage)
    ├── modelops/
    │   ├── datasets/          # Raw data files
    │   ├── models/            # Trained models
    │   ├── adapters/          # LoRA adapters
    │   └── artifacts/         # Build artifacts
    │
Delta Lake (ACID Tables)
    ├── datasets/
    │   ├── train/             # Training data (versioned)
    │   ├── validation/        # Validation data
    │   └── test/              # Test data
    │
LanceDB (Vector Storage)
    ├── embeddings/
    │   ├── documents/         # Document embeddings
    │   ├── queries/           # Query embeddings
    │   └── artifacts/         # Artifact metadata
    │
PostgreSQL (Metadata)
    ├── artifacts              # Artifact registry
    ├── jobs                   # Job tracking
    ├── datasets               # Dataset metadata
    └── deployments            # Deployment status
```

### Data Flow

1. **Ingestion**: Raw data → MinIO → Delta Lake (ACID write)
2. **Versioning**: Delta Lake maintains complete history
3. **Indexing**: Embeddings → LanceDB for search
4. **Metadata**: PostgreSQL for fast queries
5. **Analytics**: DuckDB queries Delta Lake directly

## 🔄 Workflow Orchestration

### Temporal OSS

Temporal handles all long-running workflows with:
- Automatic retries
- Workflow versioning
- Activity heartbeats
- Long-running support (24h+ jobs)

```python
@workflow.defn
class TrainingWorkflow:
    @workflow.run
    async def run(self, config):
        # Activity 1: Prepare data
        dataset = await execute_activity(prepare_dataset)
        
        # Activity 2: Train (with retries)
        model = await execute_activity(
            train_model,
            retry_policy=RetryPolicy(max_attempts=3)
        )
        
        # Activity 3: Evaluate
        metrics = await execute_activity(evaluate)
        
        # Activity 4: Register
        return await execute_activity(register_artifact)
```

### Workflow Types

1. **Dataset Ingestion** - Upload → Delta Lake → Index
2. **SFT Training** - Load → Train → Evaluate → Register
3. **Quantization** - Load → Quantize → Compare → Export
4. **RAG Setup** - Chunk → Embed → Index → Deploy
5. **RLHF** - Collect → Train Reward → PPO → Deploy

## 🎯 Service Architecture

### Microservices Design

Each service is independently scalable:

```
Training Service (GPU Workers)
    ├── DeepSpeed orchestration
    ├── Flash Attention 2
    ├── QLoRA/LoRA support
    └── Distributed training

Quantization Service (GPU Workers)
    ├── AutoAWQ
    ├── AutoGPTQ
    ├── GGUF export
    └── HQQ support

Inference Service (GPU Endpoints)
    ├── TGI (adapter serving)
    ├── vLLM (base models)
    ├── Ollama (edge)
    └── LiteLLM (proxy)

RAG Service (CPU/GPU)
    ├── Indexing (LanceDB)
    ├── Retrieval (hybrid search)
    ├── Reranking (cross-encoder)
    └── Generation (TGI)
```

### Communication

- **gRPC** for service-to-service
- **REST** for external APIs
- **WebSocket** for streaming
- **Redis** for pub/sub

## 🔐 Security Architecture

### Multi-Layer Security

1. **Transport**: TLS everywhere
2. **Authentication**: JWT tokens
3. **Authorization**: RBAC
4. **Signing**: Ed25519 for artifacts
5. **Encryption**: At rest (optional)

### Artifact Signing

```python
# Every artifact is cryptographically signed
manifest = ArtifactManifest(...)
signature = signer.sign(manifest.json())
manifest.signature = signature

# Verification before use
is_valid = signer.verify(manifest.json(), manifest.signature)
```

## 📊 Observability Stack

### Three Pillars

**Metrics (Prometheus)**
- Training job duration
- Inference latency (p50, p95, p99)
- GPU utilization
- Cache hit rates
- Error rates

**Traces (Jaeger)**
- Request flow through services
- Workflow execution paths
- Bottleneck identification
- Dependency mapping

**Logs (Loki)**
- Structured JSON logs
- Error aggregation
- Audit trails
- Debug information

### Correlation

All three are correlated via trace IDs:
```
Request ID: abc123
  ├── Trace: workflow execution
  ├── Metrics: latency, throughput
  └── Logs: detailed events
```

## 🎨 Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Control Plane                        │
├─────────────────────────────────────────────────────────┤
│  API Gateway │ Auth │ Rate Limiter │ Router            │
└────────┬────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────┐
│                  Orchestration Layer                    │
├─────────────────────────────────────────────────────────┤
│  Temporal Server │ Argo Workflows │ Job Scheduler       │
└────────┬────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────┐
│                   Service Layer                         │
├─────────────────────────────────────────────────────────┤
│  Training │ Quantization │ Inference │ RAG │ Eval       │
└────────┬────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────┐
│                    Data Layer                           │
├─────────────────────────────────────────────────────────┤
│  MinIO │ Delta Lake │ LanceDB │ PostgreSQL │ Redis      │
└────────┬────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────┐
│                 Observability Layer                     │
├─────────────────────────────────────────────────────────┤
│  Prometheus │ Grafana │ Jaeger │ Loki │ OTel Collector │
└─────────────────────────────────────────────────────────┘
```

## 🔄 Artifact Lifecycle

```
1. Development
   ├── Create dataset (Delta Lake)
   ├── Train model (Temporal workflow)
   ├── Evaluate (metrics service)
   └── Register (status: DEV)

2. Staging
   ├── Promote artifact
   ├── Integration tests
   ├── Performance tests
   └── Approval gate

3. Production
   ├── Final promotion
   ├── Deploy to inference
   ├── Monitor metrics
   └── A/B testing

4. Archived
   ├── Mark as deprecated
   ├── Retain for audit
   └── Cleanup after retention
```

## 🌐 Network Architecture

```
External
    │
    ├─── Traefik (Load Balancer)
    │       │
    │       ├─── FastAPI (Port 8000)
    │       ├─── Grafana (Port 3000)
    │       └─── MLflow (Port 5000)
    │
Internal Network (modelops)
    │
    ├─── Temporal (7233)
    ├─── PostgreSQL (5432)
    ├─── Redis (6379)
    ├─── MinIO (9000, 9001)
    ├─── Prometheus (9090)
    ├─── Jaeger (16686)
    └─── Loki (3100)
```

## 📈 Scaling Strategy

### Horizontal Scaling

- **API**: Add more replicas
- **Workers**: Auto-scale based on queue depth
- **Inference**: Scale per model demand
- **Storage**: Shard by dataset

### Vertical Scaling

- **Training**: Larger GPU instances
- **Inference**: Multi-GPU per replica
- **Database**: Increase PostgreSQL resources

## 🔌 Plugin Architecture

```python
class AlgorithmPlugin(ABC):
    @abstractmethod
    def apply(self, context):
        pass

# Custom loss
class FocalLoss(AlgorithmPlugin):
    def apply(self, context):
        return FocalLoss(alpha=0.25, gamma=2.0)

# Register
registry.register(FocalLoss())

# Use in training
loss_fn = registry.get("loss", "focal_loss").apply({})
```

## 🎯 Deployment Models

### 1. Development
- Docker Compose
- All services on single machine
- Perfect for prototyping

### 2. Production (K8s)
- Kubernetes cluster
- Separate namespaces per environment
- Auto-scaling enabled
- Multi-region support

### 3. Edge
- Ollama for inference
- GGUF quantized models
- Local vector database
- Sync with central registry

---

**This architecture provides production-grade MLOps with 100% free tools!**
