"""Simple startup script for the ModelOps API."""
import os
import sys
from pathlib import Path

# Ensure stdlib has priority over project packages (avoid 'platform' shadowing)
project_root = Path(__file__).parent
project_parent = project_root.parent
try:
    import sysconfig
    stdlib_path = sysconfig.get_paths().get("stdlib")
    if stdlib_path:
        if stdlib_path in sys.path:
            sys.path.remove(stdlib_path)
        sys.path.insert(0, stdlib_path)
except Exception:
    pass

# Add project parent first so absolute imports like 'modelops.*' resolve
if str(project_parent) not in sys.path:
    sys.path.insert(1, str(project_parent))

# Add project root at the end to avoid shadowing stdlib modules
if str(project_root) in sys.path:
    try:
        sys.path.remove(str(project_root))
    except ValueError:
        pass
sys.path.append(str(project_root))

# Create necessary directories
os.makedirs("./data", exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./datasets", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

print("=" * 60)
print("ModelOps - Lightweight Laptop Edition")
print("=" * 60)
print("\n📁 Created data directories")
print("   - ./data")
print("   - ./models")
print("   - ./datasets")
print("   - ./logs")

print("\n🚀 Starting API server...")
print("   API Docs: http://localhost:8000/docs")
print("   Health: http://localhost:8000/health")
print("   Metrics: http://localhost:8000/metrics")
print("\n" + "=" * 60 + "\n")

# Start the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.rest.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
