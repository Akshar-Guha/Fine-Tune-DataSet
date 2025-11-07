"""ModelOps SDK client."""
import os
from typing import Optional, Dict, Any, List
import requests


class ModelOpsClient:
    """Python SDK client for ModelOps platform."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize ModelOps client.
        
        Args:
            base_url: API base URL
            api_key: API key for authentication
        """
        self.base_url = base_url or os.getenv(
            "MODELOPS_API_URL",
            "http://localhost:8000"
        )
        self.api_key = api_key or os.getenv("MODELOPS_API_KEY")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}"
            })

    def create_dataset(
        self,
        name: str,
        source_path: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new dataset.
        
        Args:
            name: Dataset name
            source_path: Source file path
            description: Dataset description
            
        Returns:
            Dataset information
        """
        response = self.session.post(
            f"{self.base_url}/api/v1/datasets",
            json={
                "name": name,
                "source_path": source_path,
                "description": description
            }
        )
        response.raise_for_status()
        return response.json()

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets.
        
        Returns:
            List of datasets
        """
        response = self.session.get(f"{self.base_url}/api/v1/datasets")
        response.raise_for_status()
        return response.json()["datasets"]

    def submit_job(
        self,
        name: str,
        job_type: str,
        config: Dict[str, Any],
        dataset_id: str
    ) -> Dict[str, Any]:
        """Submit a training job.
        
        Args:
            name: Job name
            job_type: Type of job
            config: Job configuration
            dataset_id: Dataset ID
            
        Returns:
            Job information
        """
        response = self.session.post(
            f"{self.base_url}/api/v1/jobs",
            json={
                "name": name,
                "job_type": job_type,
                "config": config,
                "dataset_id": dataset_id
            }
        )
        response.raise_for_status()
        return response.json()

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job information
        """
        response = self.session.get(f"{self.base_url}/api/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def list_artifacts(
        self,
        artifact_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List artifacts.
        
        Args:
            artifact_type: Filter by artifact type
            
        Returns:
            List of artifacts
        """
        params = {}
        if artifact_type:
            params["artifact_type"] = artifact_type

        response = self.session.get(
            f"{self.base_url}/api/v1/artifacts",
            params=params
        )
        response.raise_for_status()
        return response.json()["artifacts"]

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Chat completion (OpenAI-compatible).
        
        Args:
            model: Model ID
            messages: List of messages
            **kwargs: Additional parameters
            
        Returns:
            Assistant response
        """
        response = self.session.post(
            f"{self.base_url}/api/v1/inference/chat/completions",
            json={
                "model": model,
                "messages": messages,
                **kwargs
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
