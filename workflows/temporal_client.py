"""Temporal client for workflow orchestration."""
import os
import asyncio
from typing import Dict, Any, Optional, List
import logging

from temporalio.client import Client, TLSConfig
from temporalio.worker import Worker
from temporalio.common import RetryPolicy

# Import workflows
from workflows.sft.qlora_workflow import QLoRATrainingWorkflow
from workflows.sft.config import QLoRAConfig
from workflows.quantization.awq_workflow import AWQWorkflow
from workflows.rag.indexing_workflow import RAGIndexingWorkflow

# Configure logging
logger = logging.getLogger(__name__)


class TemporalClient:
    """Client for Temporal workflow orchestration."""

    def __init__(self):
        """Initialize Temporal client."""
        # Default to local dev setup if no env vars
        self.host = os.getenv("TEMPORAL_HOST", "localhost")
        self.port = int(os.getenv("TEMPORAL_PORT", "7233"))
        self.namespace = os.getenv("TEMPORAL_NAMESPACE", "modelops")
        self.client = None
        self.worker_running = False

    async def connect(self) -> None:
        """Connect to Temporal server."""
        try:
            # Build connection URL
            connection_url = f"{self.host}:{self.port}"

            # Setup TLS if configured
            tls_config = None
            if os.getenv("TEMPORAL_TLS_ENABLED", "false").lower() == "true":
                tls_config = TLSConfig(
                    server_root_ca_cert=os.getenv("TEMPORAL_CA_CERT", ""),
                    client_cert=os.getenv("TEMPORAL_CLIENT_CERT", ""),
                    client_private_key=os.getenv("TEMPORAL_CLIENT_KEY", ""),
                )

            # Create client
            self.client = await Client.connect(
                connection_url,
                namespace=self.namespace,
                tls=tls_config,
            )
            logger.info(f"Connected to Temporal server at {connection_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Temporal server: {e}")
            raise

    async def start_worker(self) -> None:
        """Start Temporal worker to process workflows."""
        if not self.client:
            await self.connect()

        if self.worker_running:
            return

        worker = Worker(
            self.client,
            task_queue="modelops-queue",
            workflows=[
                QLoRATrainingWorkflow,
                AWQWorkflow,
                RAGIndexingWorkflow
            ],
            activities=[
                # Register all activities from workflows
                *QLoRATrainingWorkflow.get_activities(),
                *AWQWorkflow.get_activities(),
                *RAGIndexingWorkflow.get_activities()
            ],
        )

        self.worker_running = True
        await worker.run()

    async def submit_qlora_job(
        self,
        job_id: str,
        config: QLoRAConfig
    ) -> str:
        """Submit QLoRA training job.
        
        Args:
            job_id: Job ID
            config: QLoRA configuration
            
        Returns:
            Workflow execution ID
        """
        if not self.client:
            await self.connect()

        try:
            # Submit with default retry policy
            retry_policy = RetryPolicy(
                maximum_attempts=3
            )

            # Start workflow execution
            workflow_id = f"qlora-{job_id}"
            handle = await self.client.start_workflow(
                QLoRATrainingWorkflow.run,
                args=[config],
                id=workflow_id,
                task_queue="modelops-queue",
                retry_policy=retry_policy
            )
            
            logger.info(f"Started QLoRA workflow: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to submit QLoRA job: {e}")
            raise

    async def submit_quantization_job(
        self,
        job_id: str,
        config: Dict[str, Any]
    ) -> str:
        """Submit quantization job.
        
        Args:
            job_id: Job ID
            config: Quantization configuration
            
        Returns:
            Workflow execution ID
        """
        if not self.client:
            await self.connect()

        try:
            # Start workflow execution
            workflow_id = f"quantization-{job_id}"
            handle = await self.client.start_workflow(
                AWQWorkflow.run,
                args=[config],
                id=workflow_id,
                task_queue="modelops-queue",
            )
            
            logger.info(f"Started quantization workflow: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to submit quantization job: {e}")
            raise

    async def submit_rag_indexing_job(
        self,
        job_id: str,
        config: Dict[str, Any]
    ) -> str:
        """Submit RAG indexing job.
        
        Args:
            job_id: Job ID
            config: RAG indexing configuration
            
        Returns:
            Workflow execution ID
        """
        if not self.client:
            await self.connect()

        try:
            # Start workflow execution
            workflow_id = f"rag-indexing-{job_id}"
            handle = await self.client.start_workflow(
                RAGIndexingWorkflow.run,
                args=[config],
                id=workflow_id,
                task_queue="modelops-queue",
            )
            
            logger.info(f"Started RAG indexing workflow: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to submit RAG indexing job: {e}")
            raise

    async def get_workflow_status(
        self,
        workflow_id: str
    ) -> Dict[str, Any]:
        """Get workflow execution status.
        
        Args:
            workflow_id: Workflow execution ID
            
        Returns:
            Workflow status
        """
        if not self.client:
            await self.connect()
            
        try:
            handle = self.client.get_workflow_handle(workflow_id)
            desc = await handle.describe()
            
            # Build status response
            status = {
                "workflow_id": workflow_id,
                "status": desc.status.name,
                "start_time": desc.start_time,
                "close_time": desc.close_time,
                "execution_time": (desc.close_time - desc.start_time).total_seconds()
                if desc.close_time else None,
                "history_length": desc.history_length
            }
            
            return status
        
        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            raise

    async def cancel_workflow(self, workflow_id: str) -> None:
        """Cancel workflow execution.
        
        Args:
            workflow_id: Workflow execution ID
        """
        if not self.client:
            await self.connect()
            
        try:
            handle = self.client.get_workflow_handle(workflow_id)
            await handle.cancel()
            logger.info(f"Cancelled workflow: {workflow_id}")
        
        except Exception as e:
            logger.error(f"Failed to cancel workflow: {e}")
            raise


# Singleton instance
temporal_client = TemporalClient()
