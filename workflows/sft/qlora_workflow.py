"""QLoRA training workflow using Temporal."""
from datetime import timedelta
from typing import Dict, Any

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .activities import (
        prepare_dataset,
        load_base_model,
        apply_lora_config,
        train_model,
        evaluate_model,
        quantize_adapter,
        register_artifact,
        cleanup_resources
    )
    from .config import QLoRAConfig


@workflow.defn(name="qlora_training")
class QLoRATrainingWorkflow:
    """QLoRA fine-tuning workflow with DeepSpeed."""
    
    @staticmethod
    def get_activities():
        """Get activities used by this workflow."""
        return [
            prepare_dataset, 
            load_base_model, 
            apply_lora_config, 
            train_model,
            evaluate_model, 
            quantize_adapter, 
            register_artifact, 
            cleanup_resources
        ]

    @workflow.run
    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute QLoRA training workflow.
        
        Args:
            config: QLoRA training configuration as dictionary
            
        Returns:
            Training results with artifact IDs
        """
        # Convert dictionary to QLoRAConfig if needed
        if isinstance(config, dict):
            from .config import QLoRAConfig
            try:
                config = QLoRAConfig(**config)
            except Exception as e:
                workflow.logger.error(f"Failed to parse config: {str(e)}")
                raise ValueError(f"Invalid QLoRA config: {str(e)}")
        
        workflow.logger.info(f"Starting QLoRA training for {config.base_model}")

        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(minutes=5),
            maximum_attempts=3,
            backoff_coefficient=2.0
        )

        try:
            # Step 1: Prepare dataset
            workflow.logger.info("Preparing dataset...")
            dataset_info = await workflow.execute_activity(
                prepare_dataset,
                config.dataset_id,
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=retry_policy
            )

            # Step 2: Load base model
            workflow.logger.info(f"Loading base model: {config.base_model}")
            model_info = await workflow.execute_activity(
                load_base_model,
                config.base_model,
                start_to_close_timeout=timedelta(minutes=20),
                retry_policy=retry_policy
            )

            # Step 3: Apply LoRA configuration
            workflow.logger.info("Applying LoRA configuration...")
            lora_model = await workflow.execute_activity(
                apply_lora_config,
                args=[model_info, config.lora_config],
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=retry_policy
            )

            # Step 4: Train model (long-running)
            workflow.logger.info("Starting training...")
            training_results = await workflow.execute_activity(
                train_model,
                args=[lora_model, dataset_info, config.training_args],
                start_to_close_timeout=timedelta(hours=24),
                heartbeat_timeout=timedelta(minutes=5),
                retry_policy=retry_policy
            )

            # Step 5: Evaluate model
            workflow.logger.info("Evaluating model...")
            eval_metrics = await workflow.execute_activity(
                evaluate_model,
                args=[training_results["model_path"], config.eval_dataset],
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=retry_policy
            )

            # Step 6: Auto-quantize if requested
            quantized_artifact_id = None
            if config.auto_quantize:
                workflow.logger.info("Auto-quantizing adapter...")
                quantized_artifact_id = await workflow.execute_activity(
                    quantize_adapter,
                    args=[training_results["model_path"], config.quantization_method],
                    start_to_close_timeout=timedelta(hours=2),
                    retry_policy=retry_policy
                )

            # Step 7: Register artifacts
            workflow.logger.info("Registering artifacts...")
            artifact_id = await workflow.execute_activity(
                register_artifact,
                args=[training_results, eval_metrics, config],
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=retry_policy
            )

            # Step 8: Cleanup
            await workflow.execute_activity(
                cleanup_resources,
                training_results["temp_files"],
                start_to_close_timeout=timedelta(minutes=5)
            )

            workflow.logger.info(f"Training completed! Artifact ID: {artifact_id}")

            return {
                "artifact_id": artifact_id,
                "quantized_artifact_id": quantized_artifact_id,
                "metrics": eval_metrics,
                "training_duration": training_results["duration_seconds"]
            }

        except Exception as e:
            workflow.logger.error(f"Training failed: {str(e)}")
            # Attempt cleanup even on failure
            try:
                await workflow.execute_activity(
                    cleanup_resources,
                    [],
                    start_to_close_timeout=timedelta(minutes=5)
                )
            except:
                pass
            raise
