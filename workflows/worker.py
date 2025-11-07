"""Temporal worker for executing workflows."""
import asyncio
import os
from temporalio.client import Client
from temporalio.worker import Worker

# Import workflows and activities
from .sft.qlora_workflow import QLoRATrainingWorkflow
from .sft import activities as sft_activities
from .quantization.awq_workflow import AWQQuantizationWorkflow
from .rag.indexing_workflow import RAGIndexingWorkflow


async def main():
    """Run the Temporal worker."""
    temporal_url = os.getenv("TEMPORAL_URL", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "modelops-task-queue")

    # Connect to Temporal server
    client = await Client.connect(temporal_url, namespace=namespace)

    # Create worker
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[
            QLoRATrainingWorkflow,
            AWQQuantizationWorkflow,
            RAGIndexingWorkflow
        ],
        activities=[
            sft_activities.prepare_dataset,
            sft_activities.load_base_model,
            sft_activities.apply_lora_config,
            sft_activities.train_model,
            sft_activities.evaluate_model,
            sft_activities.quantize_adapter,
            sft_activities.register_artifact,
            sft_activities.cleanup_resources
        ]
    )

    print(f"🚀 Starting Temporal worker on task queue: {task_queue}")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
