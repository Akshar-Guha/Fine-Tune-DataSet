# Installation Guide - ModelOps Lightweight Edition

## Quick Install (Recommended)

### Option 1: Using pip (Fastest)

```bash
# 1. Navigate to project directory
cd "s:\projects\Fine Tunning\modelops"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API
python start_api.py

# 4. Test the API (in another terminal)
python test_api.py
```

### Option 2: Using Poetry

```bash
# 1. Install Poetry if not installed
pip install poetry

# 2. Navigate to project directory
cd "s:\projects\Fine Tunning\modelops"

# 3. Install dependencies
poetry install

# 4. Start the API
poetry run python start_api.py
```

### Option 3: Using Docker

```bash
# 1. Navigate to project directory
cd "s:\projects\Fine Tunning\modelops"

# 2. Build and start services
docker-compose up --build

# 3. Access API at http://localhost:8000/docs
```

## System Requirements

### Minimum
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.10 or 3.11
- **RAM**: 8GB
- **Disk**: 10GB free space
- **CPU**: 4 cores

### Recommended
- **OS**: Windows 11, Ubuntu 22.04, macOS 13+
- **Python**: 3.10
- **RAM**: 16GB
- **Disk**: 20GB free space
- **CPU**: 6+ cores
- **GPU**: NVIDIA GPU with 6GB+ VRAM (optional, for training)

## Troubleshooting

### Poetry Not Found
```bash
# Install Poetry
pip install poetry

# Or on Windows
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### Import Errors
```bash
# Make sure you're in the project directory
cd "s:\projects\Fine Tunning\modelops"

# Install all dependencies
pip install -r requirements.txt
```

### CUDA/GPU Issues
```bash
# For CPU-only (no GPU needed)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Port Already in Use
```bash
# Change port in start_api.py or use environment variable
# Edit start_api.py and change port=8000 to port=8001

# Or set environment variable
set API_PORT=8001  # Windows
export API_PORT=8001  # Linux/Mac
```

### Memory Issues During Training
```python
# Edit training config to use smaller batch size
config = {
    "batch_size": 1,  # Reduce from 2
    "grad_accum_steps": 16,  # Increase to compensate
    "max_seq_length": 256  # Reduce from 512
}
```

## Verify Installation

Run the test script:
```bash
python test_api.py
```

Expected output:
```
==================================================
ModelOps Lightweight Edition - API Tests
==================================================

=== Testing Health Endpoint ===
Status: 200
Response: {'status': 'healthy', 'service': 'modelops-api'}
✓ PASSED

=== Testing Root Endpoint ===
Status: 200
Response: {'message': 'ModelOps Platform API', 'version': '1.0.0', 'docs': '/docs'}
✓ PASSED

=== Testing Metrics Endpoint ===
Status: 200
Metrics available: XXX bytes
✓ PASSED

=== Testing API Documentation ===
Status: 200
Documentation available at: http://localhost:8000/docs
✓ PASSED

==================================================
Results: 4 passed, 0 failed
==================================================

🎉 All tests passed! API is working correctly.
```

## Next Steps

1. **Explore API Documentation**: http://localhost:8000/docs
2. **View Metrics**: http://localhost:8000/metrics
3. **Check Health**: http://localhost:8000/health
4. **Start MLflow**: `docker-compose up mlflow` (optional)
5. **Train a Model**: See examples in CHANGES.md

## Optional Services

### Start MLflow UI
```bash
# Using Docker
docker-compose up -d mlflow

# Or locally
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Access at: http://localhost:5000

### Start Prefect Server (Optional)
```bash
prefect server start
```

Access at: http://localhost:4200

## Getting Help

- Check CHANGES.md for detailed changes
- Review README.md for architecture overview
- Check QUICKSTART.md for usage examples
- Open API docs at http://localhost:8000/docs

## Lint Warnings

The project has some minor lint warnings (line length, whitespace) that don't affect functionality. These are cosmetic and can be safely ignored for now. Focus is on working code.
