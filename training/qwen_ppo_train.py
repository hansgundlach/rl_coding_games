"""Qwen model PPO training pipeline for Connect-X agent."""

import os
import sys
import yaml
import argparse
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.wandb_logger import WandBLogger
from training.lora_utils import (
    setup_lora_model,
    save_lora_checkpoint,
    prepare_model_for_training,
)
from environment.connectx_wrapper import ConnectXWrapper
from environment.reward_shaping import RewardShaper
from agents.qwen_policy import QwenPolicyAgent
from evaluation.evalplus_runner import EvalPlusRunner
from utils.env_loader import get_api_key
import wandb


class QwenPPOExperience:
    """Experience tuple for Qwen PPO training."""

    def __init__(
        self, state, action, reward, next_state, done, log_prob, prompt, response
    ):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.log_prob = log_prob
        self.prompt = prompt
        self.response = response


class QwenConnectXPPOTrainer:
    """PPO trainer for Qwen model on Connect-X."""

    def __init__(self, config_path: str):
        """Initialize Qwen PPO trainer."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        print(f"Loaded config from {config_path}")
        self.setup_logging()
        self.setup_environment()
        self.setup_qwen_agent()
        self.experience_buffer = []

    def setup_logging(self):
        """Initialize Weights & Biases logging."""
        wandb_key = get_api_key("wandb", required=False)
        if wandb_key:
            wandb.login(key=wandb_key)

        self.logger = WandBLogger(
            project=self.config["logging"]["project"],
            entity=self.config["logging"].get("entity"),
            name=f"qwen-connectx-ppo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=self.config,
        )

    def setup_environment(self):
        """Setup Connect-X environment."""
        env_config = self.config["environment"]
        reward_shaper = (
            RewardShaper() if env_config.get("reward_shaping", False) else None
        )

        self.env = ConnectXWrapper(
            opponent_agent="random",
            reward_shaper=reward_shaper,
            rows=env_config["rows"],
            columns=env_config["columns"],
        )
        print("Environment setup completed")

    def setup_qwen_agent(self):
        """Setup Qwen agent with LoRA."""
        qwen_config = self.config["qwen"]
        lora_config = self.config["lora"]

        # Setup LoRA model
        self.model = setup_lora_model(
            model_name=qwen_config["model_name"],
            lora_config=lora_config,
            device=qwen_config.get("device", "auto"),
            cache_dir=qwen_config.get("cache_dir", None),
            local_files_only=qwen_config.get("local_files_only", False),
        )

        # Prepare model for training
        prepare_model_for_training(self.model)

        # Setup tokenizer
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(qwen_config["model_name"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create Qwen agent (for inference)
        self.agent = QwenPolicyAgent(
            model_name=qwen_config["model_name"],
            device=qwen_config.get("device", "auto"),
            max_new_tokens=qwen_config.get("max_new_tokens", 10),
            temperature=qwen_config.get("temperature", 0.7),
        )

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"].get("weight_decay", 0.01)),
        )

        print("Qwen agent setup completed")

    def collect_episode(self) -> tuple:
        """Collect one episode of experience with Qwen model."""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        experiences = []

        while True:
            # Convert state to board format
            board = np.array(state).reshape(
                self.config["environment"]["rows"],
                self.config["environment"]["columns"],
            )

            # Get valid actions
            valid_actions = self.agent.get_valid_actions(board)
            if not valid_actions:
                break

            # Generate prompt for current state
            total_pieces = np.sum(board > 0)
            player_mark = 1 if total_pieces % 2 == 0 else 2
            prompt = self.agent._create_game_prompt(board, player_mark)

            # Get action from Qwen agent
            action = self.agent.act(
                {"board": state.flatten().tolist()},
                {
                    "rows": self.config["environment"]["rows"],
                    "columns": self.config["environment"]["columns"],
                },
            )

            # Take step in environment
            next_state, reward, done, info = self.env.step(action)

            # Generate response text (for training)
            response = str(action)

            # Compute log probability (simplified for now)
            log_prob = torch.log(torch.tensor(1.0 / len(valid_actions)))

            # Store experience
            experience = QwenPPOExperience(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=done,
                log_prob=log_prob,
                prompt=prompt,
                response=response,
            )
            experiences.append(experience)

            episode_reward += reward
            episode_length += 1

            state = next_state

            if done or episode_length >= 100:  # Max episode length
                break

        return experiences, episode_reward, episode_length

    def compute_rewards_to_go(
        self, experiences: List[QwenPPOExperience]
    ) -> List[float]:
        """Compute discounted rewards-to-go."""
        gamma = float(self.config["ppo"]["gamma"])
        rewards_to_go = []
        running_return = 0

        for exp in reversed(experiences):
            running_return = exp.reward + gamma * running_return
            rewards_to_go.insert(0, running_return)

        return rewards_to_go

    def update_qwen_model(
        self, batch_experiences: List[QwenPPOExperience]
    ) -> Dict[str, float]:
        """Update Qwen model using PPO-style loss."""
        if not batch_experiences:
            return {}

        # Compute rewards-to-go
        rewards_to_go = self.compute_rewards_to_go(batch_experiences)

        # Prepare training data
        prompts = [exp.prompt for exp in batch_experiences]
        responses = [exp.response for exp in batch_experiences]
        rewards = torch.tensor(rewards_to_go, dtype=torch.float32)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        total_loss = 0
        num_batches = 0

        # Mini-batch training
        batch_size = int(self.config["training"]["batch_size"])
        for i in range(0, len(batch_experiences), batch_size):
            mini_batch_prompts = prompts[i : i + batch_size]
            mini_batch_responses = responses[i : i + batch_size]
            mini_batch_rewards = rewards[i : i + batch_size]

            # Tokenize prompts and responses
            full_texts = [
                f"{prompt}{response}"
                for prompt, response in zip(mini_batch_prompts, mini_batch_responses)
            ]

            inputs = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            if self.model.device != torch.device("cpu"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                mini_batch_rewards = mini_batch_rewards.to(self.model.device)

            # Forward pass
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Weight loss by rewards (higher reward = lower loss)
            weighted_loss = loss * (1.0 - mini_batch_rewards.mean())

            # Backward pass
            self.optimizer.zero_grad()
            weighted_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(self.config["training"]["max_grad_norm"])
            )

            self.optimizer.step()

            total_loss += weighted_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        return {
            "policy_loss": avg_loss,
            "num_batches": num_batches,
        }

    def evaluate_coding_performance(
        self, checkpoint_path: str = None, force_full_eval: bool = False
    ) -> Dict[str, Any]:
        """Evaluate model on coding benchmarks."""
        try:
            eval_config = self.config.get("evaluation", {})

            # Determine evaluation mode
            use_quick_eval = eval_config.get("quick_eval", True) and not force_full_eval
            eval_mode = "Quick" if use_quick_eval else "Full"
            print(f"Running {eval_mode.lower()} coding benchmark evaluation...")

            # Use current model or specified checkpoint
            model_path = checkpoint_path or self.config["qwen"]["model_name"]

            runner = EvalPlusRunner(
                model_path=model_path,
                lora_path=checkpoint_path,
                device=self.config["qwen"].get("device", "auto"),
                max_new_tokens=512,
                temperature=0.2,
                num_samples=eval_config.get("num_coding_samples", 1),
                quick_eval=use_quick_eval,
                quick_eval_size=eval_config.get("quick_eval_size", 10),
            )

            results = runner.run_evaluation()
            print(f"Coding evaluation results ({eval_mode}): {results}")

            return results

        except Exception as e:
            print(f"Coding evaluation failed: {e}")
            return {
                "humaneval_plus": {"pass@1": 0.0, "pass@10": 0.0},
                "mbpp_plus": {"pass@1": 0.0, "pass@10": 0.0},
                "eval_mode": "Failed",
                "total_eval_time_seconds": 0,
            }

    def train(self):
        """Run Qwen PPO training loop."""
        print("Starting Qwen PPO training...")
        training_config = self.config["training"]

        total_steps = 0

        # Run initial coding evaluation (skip for faster startup)
        if self.config.get("evaluation", {}).get("skip_initial_eval", False):
            print("Skipping initial coding evaluation...")
        else:
            initial_coding_results = self.evaluate_coding_performance()
            self.logger.log_coding_evaluation(initial_coding_results, 0)

        for epoch in range(training_config["num_epochs"]):
            print(f"Epoch {epoch + 1}/{training_config['num_epochs']}")

            # Collect episodes
            epoch_rewards = []
            epoch_lengths = []
            epoch_experiences = []

            for episode in range(training_config["episodes_per_epoch"]):
                experiences, episode_reward, episode_length = self.collect_episode()

                epoch_experiences.extend(experiences)
                epoch_rewards.append(episode_reward)
                epoch_lengths.append(episode_length)
                total_steps += episode_length

                if (episode + 1) % 5 == 0:
                    print(
                        f"  Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}"
                    )

            # Update Qwen model
            if epoch_experiences:
                update_stats = self.update_qwen_model(epoch_experiences)

                # Log training statistics
                stats = {
                    "qwen_ppo/loss/policy": update_stats.get("policy_loss", 0),
                    "qwen_ppo/num_batches": update_stats.get("num_batches", 0),
                    "episode/mean_reward": np.mean(epoch_rewards),
                    "episode/mean_length": np.mean(epoch_lengths),
                    "episode/min_reward": np.min(epoch_rewards),
                    "episode/max_reward": np.max(epoch_rewards),
                    "training/total_steps": total_steps,
                }

                self.logger.log_training_stats(stats, epoch)

                print(
                    f"  Mean reward: {np.mean(epoch_rewards):.2f} ± {np.std(epoch_rewards):.2f}"
                )
                print(f"  Policy loss: {update_stats.get('policy_loss', 0):.4f}")

            # Save checkpoint and evaluate
            if (epoch + 1) % self.config["checkpoints"].get("save_interval", 5) == 0:
                checkpoint_dir = self.config["checkpoints"]["save_dir"]
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"qwen_ppo_epoch_{epoch + 1}"
                )

                # Save LoRA checkpoint
                save_lora_checkpoint(self.model, checkpoint_path)
                print(f"  Saved LoRA checkpoint: {checkpoint_path}")

                # Determine evaluation type
                eval_config = self.config.get("evaluation", {})
                full_eval_interval = eval_config.get("full_eval_interval", 0)

                # Run full evaluation if specified interval is reached
                if full_eval_interval > 0 and (epoch + 1) % full_eval_interval == 0:
                    print(f"  Running full evaluation at epoch {epoch + 1}")
                    coding_results = self.evaluate_coding_performance(
                        checkpoint_path, force_full_eval=True
                    )
                else:
                    # Run quick evaluation by default
                    coding_results = self.evaluate_coding_performance(
                        checkpoint_path, force_full_eval=False
                    )

                self.logger.log_coding_evaluation(coding_results, epoch + 1)

        print("Training completed!")
        self.logger.finish()


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Qwen Connect-X PPO agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen_ppo.yaml",
        help="Path to training configuration file",
    )

    args = parser.parse_args()

    trainer = QwenConnectXPPOTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
