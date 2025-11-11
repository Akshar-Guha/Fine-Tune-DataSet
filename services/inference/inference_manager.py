"""Inference Manager - Deploy and manage inference endpoints."""
from typing import Dict, Any, Optional, List
from pathlib import Path
import subprocess
import requests
import time
import json


class InferenceManager:
    """Manage model inference deployments."""
    
    def __init__(self):
        """Initialize inference manager."""
        self.deployments = {}
        
    async def deploy_ollama(
        self,
        model_path: str,
        port: int = 11434,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Deploy model using Ollama.
        
        Args:
            model_path: Path to GGUF model
            port: Ollama server port
            model_name: Custom model name
            
        Returns:
            Deployment information
        """
        try:
            # Check if Ollama is running
            try:
                response = requests.get(f"http://localhost:{port}/api/tags")
                ollama_running = response.status_code == 200
            except:
                ollama_running = False
            
            if not ollama_running:
                print("Starting Ollama server...")
                # Start Ollama in background
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(3)
            
            # Create Modelfile
            if model_name is None:
                model_name = Path(model_path).stem
            
            modelfile_content = f"""FROM {model_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
            
            modelfile_path = Path(model_path).parent / "Modelfile"
            with open(modelfile_path, "w") as f:
                f.write(modelfile_content)
            
            # Create model in Ollama
            print(f"Creating Ollama model: {model_name}")
            create_cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
            result = subprocess.run(create_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Model creation failed: {result.stderr}")
            
            # Test inference
            test_result = self.test_ollama_inference(model_name, port)
            
            deployment_info = {
                "backend": "ollama",
                "model_name": model_name,
                "port": port,
                "endpoint": f"http://localhost:{port}",
                "api_endpoint": f"http://localhost:{port}/api/generate",
                "status": "running",
                "test_result": test_result
            }
            
            self.deployments[model_name] = deployment_info
            
            print(f"✓ Ollama deployment complete!")
            print(f"  API: http://localhost:{port}/api/generate")
            print(f"  Model: {model_name}")
            
            return deployment_info
            
        except Exception as e:
            return {
                "backend": "ollama",
                "status": "failed",
                "error": str(e)
            }
    
    def test_ollama_inference(
        self,
        model_name: str,
        port: int = 11434
    ) -> Dict[str, Any]:
        """Test Ollama model inference.
        
        Args:
            model_name: Name of the model
            port: Ollama port
            
        Returns:
            Test results
        """
        try:
            url = f"http://localhost:{port}/api/generate"
            payload = {
                "model": model_name,
                "prompt": "Hello, how are you?",
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "tokens": result.get("eval_count", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def deploy_tgi(
        self,
        model_path: str,
        port: int = 8080,
        num_shard: int = 1
    ) -> Dict[str, Any]:
        """Deploy model using Text Generation Inference (TGI).
        
        Args:
            model_path: Path to model
            port: TGI server port
            num_shard: Number of shards for model parallelism
            
        Returns:
            Deployment information
        """
        try:
            # Start TGI container
            print(f"Starting TGI server for {model_path}...")
            
            docker_cmd = [
                "docker", "run", "-d",
                "--name", f"tgi-{port}",
                "-p", f"{port}:80",
                "--gpus", "all",
                "-v", f"{Path(model_path).parent}:/data",
                "ghcr.io/huggingface/text-generation-inference:latest",
                "--model-id", str(model_path),
                "--num-shard", str(num_shard)
            ]
            
            result = subprocess.run(docker_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"TGI deployment failed: {result.stderr}")
            
            container_id = result.stdout.strip()
            
            # Wait for server to be ready
            print("Waiting for TGI server to start...")
            max_retries = 30
            for i in range(max_retries):
                try:
                    response = requests.get(f"http://localhost:{port}/health")
                    if response.status_code == 200:
                        break
                except:
                    pass
                time.sleep(2)
            
            deployment_info = {
                "backend": "tgi",
                "container_id": container_id,
                "port": port,
                "endpoint": f"http://localhost:{port}",
                "api_endpoint": f"http://localhost:{port}/generate",
                "status": "running",
                "num_shard": num_shard
            }
            
            self.deployments[f"tgi-{port}"] = deployment_info
            
            print(f"✓ TGI deployment complete!")
            print(f"  API: http://localhost:{port}/generate")
            
            return deployment_info
            
        except Exception as e:
            return {
                "backend": "tgi",
                "status": "failed",
                "error": str(e)
            }
    
    async def deploy_vllm(
        self,
        model_path: str,
        port: int = 8000,
        tensor_parallel_size: int = 1
    ) -> Dict[str, Any]:
        """Deploy model using vLLM.
        
        Args:
            model_path: Path to model
            port: vLLM server port
            tensor_parallel_size: Tensor parallelism degree
            
        Returns:
            Deployment information
        """
        try:
            print(f"Starting vLLM server for {model_path}...")
            
            # Start vLLM server
            vllm_cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(model_path),
                "--port", str(port),
                "--tensor-parallel-size", str(tensor_parallel_size)
            ]
            
            process = subprocess.Popen(
                vllm_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            print("Waiting for vLLM server to start...")
            time.sleep(10)
            
            deployment_info = {
                "backend": "vllm",
                "process_id": process.pid,
                "port": port,
                "endpoint": f"http://localhost:{port}",
                "api_endpoint": f"http://localhost:{port}/v1/completions",
                "status": "running",
                "tensor_parallel_size": tensor_parallel_size
            }
            
            self.deployments[f"vllm-{port}"] = deployment_info
            
            print(f"✓ vLLM deployment complete!")
            print(f"  API: http://localhost:{port}/v1/completions")
            
            return deployment_info
            
        except Exception as e:
            return {
                "backend": "vllm",
                "status": "failed",
                "error": str(e)
            }
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all active deployments.
        
        Returns:
            List of deployment information
        """
        return list(self.deployments.values())
    
    async def stop_deployment(self, deployment_id: str) -> bool:
        """Stop a deployment.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            True if stopped successfully
        """
        if deployment_id not in self.deployments:
            return False
        
        deployment = self.deployments[deployment_id]
        backend = deployment["backend"]
        
        try:
            if backend == "tgi":
                # Stop Docker container
                subprocess.run(
                    ["docker", "stop", deployment["container_id"]],
                    check=True
                )
                subprocess.run(
                    ["docker", "rm", deployment["container_id"]],
                    check=True
                )
            elif backend == "vllm":
                # Kill process
                import os
                import signal
                os.kill(deployment["process_id"], signal.SIGTERM)
            elif backend == "ollama":
                # Ollama models stay loaded, just remove from tracking
                pass
            
            del self.deployments[deployment_id]
            print(f"✓ Stopped deployment: {deployment_id}")
            return True
            
        except Exception as e:
            print(f"Error stopping deployment: {e}")
            return False
