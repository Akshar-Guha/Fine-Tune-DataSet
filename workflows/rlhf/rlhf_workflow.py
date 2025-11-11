"""RLHF Workflow - Reinforcement Learning from Human Feedback."""
from datetime import timedelta
from typing import Dict, Any
from temporalio import workflow
from temporalio.common import RetryPolicy


@workflow.defn(name="rlhf_training")
class RLHFTrainingWorkflow:
    """RLHF workflow for preference-based training."""
    
    @workflow.run
    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RLHF training workflow.
        
        Args:
            config: RLHF configuration
            
        Returns:
            Training results
        """
        workflow.logger.info("Starting RLHF training workflow...")
        
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(minutes=5),
            maximum_attempts=3,
            backoff_coefficient=2.0
        )
        
        # Step 1: Train reward model (PPO)
        workflow.logger.info("Training reward model...")
        reward_model = await workflow.execute_activity(
            "train_reward_model",
            config["reward_config"],
            start_to_close_timeout=timedelta(hours=12),
            retry_policy=retry_policy
        )
        
        # Step 2: Generate responses
        workflow.logger.info("Generating responses for preference learning...")
        responses = await workflow.execute_activity(
            "generate_responses",
            args=[config["base_model"], config["prompts"]],
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=retry_policy
        )
        
        # Step 3: Collect human feedback (or simulated)
        workflow.logger.info("Collecting feedback...")
        feedback = await workflow.execute_activity(
            "collect_feedback",
            responses,
            start_to_close_timeout=timedelta(hours=24),
            retry_policy=retry_policy
        )
        
        # Step 4: Train policy with PPO
        workflow.logger.info("Training policy with PPO...")
        policy_results = await workflow.execute_activity(
            "train_ppo_policy",
            args=[config["base_model"], reward_model, feedback],
            start_to_close_timeout=timedelta(hours=24),
            retry_policy=retry_policy
        )
        
        # Step 5: Evaluate aligned model
        workflow.logger.info("Evaluating aligned model...")
        evaluation = await workflow.execute_activity(
            "evaluate_rlhf_model",
            policy_results["model_path"],
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=retry_policy
        )
        
        workflow.logger.info("RLHF training complete!")
        
        return {
            "reward_model": reward_model,
            "policy_model": policy_results["model_path"],
            "evaluation": evaluation,
            "training_duration": policy_results.get("duration_seconds", 0)
        }
