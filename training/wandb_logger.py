"""Weights & Biases logging utilities."""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import wandb


class WandBLogger:
    """Weights & Biases logger for Connect-X training."""
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize W&B logger."""
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config
        self.step = 0
        
        # Initialize wandb run
        wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            reinit=True
        )
        print(f"W&B: Initialized project {project}")
    
    def log_training_stats(self, stats: Dict[str, float], epoch: int):
        """Log PPO training statistics."""
        wandb.log({**stats, "epoch": epoch}, step=self.step)
        print(f"W&B: Logged training stats for epoch {epoch}")
        self.step += 1
    
    def log_evaluation_results(self, results: Dict[str, float], tag: str):
        """Log evaluation benchmark results."""
        prefixed_results = {f"{tag}/{k}": v for k, v in results.items()}
        wandb.log(prefixed_results)
        print(f"W&B: Logged evaluation results with tag {tag}")
    
    def log_game_visualization(self, board_states: list, actions: list, rewards: list):
        """Log Connect-X game visualization."""
        # Log game data as tables for now
        wandb.log({
            "game/num_moves": len(actions),
            "game/total_reward": sum(rewards),
            "game/final_reward": rewards[-1] if rewards else 0
        })
        print("W&B: Logged game visualization")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log custom metrics."""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        wandb.log(metrics)
        print(f"W&B: Logged metrics with prefix {prefix}")
    
    def log_coding_evaluation(self, results: Dict[str, Any], epoch: int):
        """Log coding benchmark evaluation results."""
        coding_metrics = {}
        
        # Log HumanEval-Plus results
        if 'humaneval_plus' in results:
            humaneval_results = results['humaneval_plus']
            coding_metrics.update({
                "coding/humaneval_plus/pass@1": humaneval_results.get('pass@1', 0.0),
                "coding/humaneval_plus/pass@10": humaneval_results.get('pass@10', 0.0),
            })
        
        # Log MBPP-Plus results
        if 'mbpp_plus' in results:
            mbpp_results = results['mbpp_plus']
            coding_metrics.update({
                "coding/mbpp_plus/pass@1": mbpp_results.get('pass@1', 0.0),
                "coding/mbpp_plus/pass@10": mbpp_results.get('pass@10', 0.0),
            })
        
        # Add epoch information
        coding_metrics["epoch"] = epoch
        
        wandb.log(coding_metrics, step=self.step)
        print(f"W&B: Logged coding evaluation results for epoch {epoch}")
        
        # Also create a summary table
        if epoch > 0:  # Don't create table for initial evaluation
            summary_data = []
            for benchmark, benchmark_results in results.items():
                if isinstance(benchmark_results, dict):
                    for metric, value in benchmark_results.items():
                        summary_data.append([epoch, benchmark, metric, value])
            
            if summary_data:
                table = wandb.Table(
                    columns=["Epoch", "Benchmark", "Metric", "Score"],
                    data=summary_data
                )
                wandb.log({"coding_evaluation_summary": table}, step=self.step)
    
    def finish(self):
        """Finish W&B run."""
        wandb.finish()
        print("W&B: Finished run")