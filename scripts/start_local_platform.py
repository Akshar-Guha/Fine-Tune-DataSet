"""Start the complete local LLM fine-tuning platform."""
import subprocess
import sys
import time
import os
from pathlib import Path


def print_banner():
    """Print startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║         🚀 LOCAL LLM FINE-TUNING PLATFORM 🚀                 ║
    ║                                                               ║
    ║   Complete platform for local LLM fine-tuning with RLHF      ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("datasets", "Datasets"),
        ("fastapi", "FastAPI"),
    ]
    
    missing = []
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Run: poetry install")
        return False
    
    print("✅ All dependencies installed!\n")
    return True


def create_directories():
    """Create required directories."""
    print("📁 Creating directories...")
    
    dirs = [
        "models",
        "datasets",
        "training_output",
        "quantized",
        "logs",
        "data"
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_name}/")
    
    print()


def start_mlflow():
    """Start MLflow tracking server."""
    print("🎯 Starting MLflow...")
    
    try:
        mlflow_process = subprocess.Popen(
            [sys.executable, "-m", "mlflow", "server", 
             "--host", "0.0.0.0", 
             "--port", "5000",
             "--backend-store-uri", "sqlite:///mlflow.db",
             "--default-artifact-root", "./mlflow_artifacts"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        time.sleep(3)
        print("  ✓ MLflow running on http://localhost:5000\n")
        return mlflow_process
    except Exception as e:
        print(f"  ⚠ MLflow failed to start: {e}\n")
        return None


def start_api():
    """Start FastAPI server."""
    print("🌐 Starting API server...")
    
    try:
        api_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", 
             "api.rest.main:app",
             "--host", "0.0.0.0",
             "--port", "8000",
             "--reload"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        time.sleep(3)
        print("  ✓ API running on http://localhost:8000\n")
        print("  📖 API Docs: http://localhost:8000/docs\n")
        return api_process
    except Exception as e:
        print(f"  ⚠ API failed to start: {e}\n")
        return None


def start_frontend():
    """Start frontend development server."""
    print("🎨 Starting frontend...")
    
    frontend_dir = Path("frontend")
    
    if not frontend_dir.exists():
        print("  ⚠ Frontend not found (optional)\n")
        return None
    
    try:
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print("  Installing frontend dependencies...")
            subprocess.run(
                ["npm", "install"],
                cwd=frontend_dir,
                check=True,
                stdout=subprocess.DEVNULL
            )
        
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        time.sleep(5)
        print("  ✓ Frontend running on http://localhost:3000\n")
        return frontend_process
    except Exception as e:
        print(f"  ⚠ Frontend failed to start: {e}\n")
        return None


def print_quick_start():
    """Print quick start guide."""
    guide = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                      QUICK START GUIDE                        ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    🌐 Access Points:
       • API Documentation: http://localhost:8000/docs
       • Frontend Dashboard: http://localhost:3000
       • MLflow Tracking: http://localhost:5000
    
    📚 Example Usage (Python):
    
    from services.model_registry import ModelRegistry
    from services.dataset_registry import DatasetRegistry
    from services.training_orchestrator import TrainingOrchestrator
    
    # Download model
    model_reg = ModelRegistry()
    model_path = model_reg.download_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Download dataset
    dataset_reg = DatasetRegistry()
    dataset_path = dataset_reg.download_dataset("timdettmers/openassistant-guanaco")
    
    # Fine-tune
    orchestrator = TrainingOrchestrator()
    config = {
        "lora_rank": 8,
        "num_epochs": 3,
        "batch_size": 2,
        "learning_rate": 2e-4
    }
    
    results = await orchestrator.execute_qlora_training(
        "my_job",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "timdettmers/openassistant-guanaco",
        config
    )
    
    print(f"Training complete! Perplexity: {results['perplexity']}")
    
    📖 Full Documentation:
       See LOCAL_FINETUNING_COMPLETE_GUIDE.md
    
    ⏸ To Stop:
       Press Ctrl+C
    
    ╔═══════════════════════════════════════════════════════════════╗
    ║                 PLATFORM READY! 🚀                           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(guide)


def main():
    """Main startup function."""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Start services
    processes = []
    
    mlflow = start_mlflow()
    if mlflow:
        processes.append(mlflow)
    
    api = start_api()
    if api:
        processes.append(api)
    
    frontend = start_frontend()
    if frontend:
        processes.append(frontend)
    
    # Print guide
    print_quick_start()
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        for process in processes:
            process.terminate()
        print("✅ All services stopped.\n")


if __name__ == "__main__":
    main()
