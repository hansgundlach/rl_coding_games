#!/usr/bin/env python3
"""
SPIRAL Iterated Prisoner's Dilemma Training

Self-play training where LLMs compete by submitting code strategies for the
iterated prisoner's dilemma. Players must write strategy functions that decide
whether to cooperate or defect based on game history.

Key Features:
- LLMs submit code strategies, not direct actions
- Reward based on tournament wins/losses
- -1 reward for code execution failures
- Extensible game framework for other strategy games
"""

# Set environment variable to avoid tokenizers warning
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import wandb
import sys
import os
import datetime
import glob
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import json
import yaml
import argparse

print("🚀 Starting SPIRAL Prisoner's Dilemma Training...")
print("📋 Initializing components...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="SPIRAL Prisoner's Dilemma Training")
parser.add_argument(
    "--config",
    type=str,
    default="configs/spiral_prisoners_dilemma.yaml",
    help="Path to configuration file",
)

# Parse known args to allow config overrides
args, unknown_args = parser.parse_known_args()

# Load configuration
print(f"📝 Loading configuration from: {args.config}")
with open(args.config, "r") as f:
    config = yaml.safe_load(f)


# Apply config overrides from command line
def apply_config_overrides(config, override_args):
    """Apply command line config overrides to loaded config."""
    overrides_applied = []

    i = 0
    while i < len(override_args):
        arg = override_args[i]

        # Handle --key=value format
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.lstrip("-")
        # Handle --key value format
        elif arg.startswith("--") and i + 1 < len(override_args):
            key = arg.lstrip("-")
            value = override_args[i + 1]
            i += 1  # Skip next arg since we used it as value
        else:
            i += 1
            continue

        # Convert value to appropriate type
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        elif (
            value.replace(".", "", 1)
            .replace("e-", "", 1)
            .replace("e+", "", 1)
            .isdigit()
        ):
            value = float(value)

        # Apply override to config using dot notation
        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        old_value = current.get(keys[-1], "NOT_SET")
        current[keys[-1]] = value
        overrides_applied.append(f"{key}: {old_value} -> {value}")

        i += 1

    if overrides_applied:
        print("🔧 Applied config overrides:")
        for override in overrides_applied:
            print(f"   {override}")

    return config


# Apply any config overrides
config = apply_config_overrides(config, unknown_args)

# Extract config values for easy access
WANDB_ENABLED = config["wandb"]["enabled"]


def detect_platform_and_gpu():
    """Auto-detect platform and GPU capabilities for environment-specific settings."""
    print("🔍 Detecting platform and GPU capabilities...")
    if not torch.cuda.is_available():
        return {
            "supports_bf16": False,
            "device": "cpu",
            "gpu_type": "none",
            "offline_mode": False,
            "platform": "cpu",
        }

    # Get GPU name for detection
    gpu_name = torch.cuda.get_device_name(0).upper()
    hostname = os.uname().nodename.lower()
    current_path = os.getcwd().lower()

    # Platform detection logic
    is_supercloud = (
        "gridsan" in hostname
        or "supercloud" in hostname
        or "/home/gridsan/" in current_path
    )
    is_lambda = "lambda" in hostname or "/lambda/" in current_path

    # GPU type detection
    if "V100" in gpu_name:
        gpu_type = "V100"
        supports_bf16 = False
    elif "A100" in gpu_name or "H100" in gpu_name:
        gpu_type = "A100+" if "A100" in gpu_name else "H100"
        supports_bf16 = True
    else:
        gpu_type = "unknown"
        supports_bf16 = False

    # Offline mode detection
    if is_supercloud:
        offline_mode = True
        offline_reason = "detected Supercloud environment"
    else:
        try:
            import socket

            socket.create_connection(("8.8.8.8", 53), timeout=3)
            offline_mode = False
            offline_reason = "internet connection available"
        except (socket.error, socket.timeout):
            offline_mode = True
            offline_reason = "no internet connection detected"

    platform = "Supercloud" if is_supercloud else ("Lambda" if is_lambda else "unknown")

    return {
        "supports_bf16": supports_bf16,
        "device": "cuda",
        "gpu_type": gpu_type,
        "offline_mode": offline_mode,
        "platform": platform,
        "offline_reason": offline_reason,
    }


# Auto-detect platform and capabilities
print("🔧 Running platform detection...")
platform_info = detect_platform_and_gpu()
print(f"🔍 Auto-detected: {platform_info['platform']} platform")
print(
    f"🎮 GPU: {platform_info['gpu_type']}, BF16 support: {platform_info['supports_bf16']}"
)
print(
    f"🌐 Offline mode: {platform_info['offline_mode']} ({platform_info['offline_reason']})"
)

# Extract values for easier use
offline_mode = platform_info["offline_mode"]

# Set global environment variables for transformers library and W&B
if WANDB_ENABLED:  # Check if W&B is enabled by user
    if offline_mode:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["WANDB_MODE"] = "offline"
        print("✅ Set global offline mode for transformers and wandb")
    else:
        # For online mode, ensure offline flags are not set
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        os.environ.pop("WANDB_MODE", None)
else:  # If WANDB_ENABLED is False, explicitly disable W&B
    os.environ["WANDB_MODE"] = "disabled"
    print("🚫 W&B logging explicitly disabled by user configuration.")

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("📦 Loading utility modules...")
from utils.env_loader import get_api_key
from evaluation.mbpp.evaluator import MBPPEvaluator, EvalConfig
from game_environments.prisoners_dilemma import IteratedPrisonersDilemma
from game_environments.base_game import PlayerSubmission

# Initialize wandb with API key from environment (skip if W&B is not enabled)
if WANDB_ENABLED:  # Only try to log in if W&B is enabled
    wandb_key = get_api_key("wandb", required=False)
    if wandb_key:
        wandb.login(key=wandb_key)
        print("✓ Logged into W&B using environment variable")
    else:
        print(
            "⚠️ No W&B API key found, continuing with W&B logging (if online) or local saves (if offline)."
        )
else:
    print("🚫 Skipping W&B login (W&B is disabled by user).")

# Initialize MBPP evaluator with consolidated config
print("🧪 Setting up MBPP evaluator...")

# Create evaluation config from main config
eval_config_dict = config.get("evaluation", {}).copy()

# Remove keys not expected by EvalConfig constructor
eval_config_dict.pop("enabled_initial", None)
eval_config_dict.pop("enabled_final", None)
eval_config_dict.pop("enabled_interval", None)
eval_config_dict.pop("eval_interval_steps", None)

# Create EvalConfig object from consolidated config
eval_config = EvalConfig(**eval_config_dict)

# Print evaluation config summary
print("\n" + "=" * 50)
print("🧪 MBPP Evaluation Configuration")
print("=" * 50)
print(f"Enabled: {'✅' if eval_config.enabled else '❌'}")
if eval_config.enabled:
    print(f"Questions: {eval_config.num_questions}")
    print(
        f"Initial eval: {'✅' if config['evaluation'].get('enabled_initial', True) else '❌'}"
    )
    print(
        f"Final eval: {'✅' if config['evaluation'].get('enabled_final', True) else '❌'}"
    )
    print(
        f"Interval eval: {'✅' if config['evaluation'].get('enabled_interval', False) else '❌'}"
    )
    if config["evaluation"].get("enabled_interval", False):
        print(
            f"Eval every: {config['evaluation'].get('eval_interval_steps', 'N/A')} steps"
        )
    print(f"Dataset: {eval_config.dataset_path or 'auto-detect'}")
    print(f"Results dir: {eval_config.results_dir}")
    print(f"Temperature: {eval_config.temperature}")
    print(f"Max tokens: {eval_config.max_new_tokens}")
    print(f"Timeout: {eval_config.timeout_seconds}s")
print("=" * 50 + "\n")

mbpp_evaluator = MBPPEvaluator(eval_config)

if not mbpp_evaluator.config.enabled:
    print("⚠️ MBPP evaluation disabled - dataset not found")
    print("💡 To enable evaluation, download MBPP dataset first:")
    print("   python -m evaluation.mbpp.evaluator")
else:
    print(
        f"✅ MBPP evaluation enabled with {mbpp_evaluator.config.num_questions} questions"
    )


@dataclass
class StrategyGameTrajectory:
    """Store trajectory data for a single strategy game."""

    player1_data: Dict  # Player 1's submission and trajectory
    player2_data: Dict  # Player 2's submission and trajectory
    game_outcome: Dict  # Final results and rewards
    game_result: Any  # Full game result from the environment


class RoleConditionedAdvantageEstimation:
    """
    Role-conditioned Advantage Estimation (RAE) from SPIRAL paper.
    Maintains separate baselines for each player role to reduce variance.
    """

    def __init__(self, alpha: float = 0.95):
        self.alpha = alpha  # EMA decay rate
        self.baselines = {"player1": 0.0, "player2": 0.0}
        self.update_counts = {"player1": 0, "player2": 0}

    def update_baseline(self, role: str, reward: float):
        """Update baseline for specific player role using EMA."""
        if self.update_counts[role] == 0:
            # First update
            self.baselines[role] = reward
        else:
            # EMA update
            self.baselines[role] = (
                self.alpha * self.baselines[role] + (1 - self.alpha) * reward
            )
        self.update_counts[role] += 1

    def compute_advantage(self, role: str, reward: float) -> float:
        """Compute advantage for player by subtracting role-specific baseline."""
        return reward - self.baselines[role]

    def get_stats(self) -> Dict:
        """Get baseline statistics for logging."""
        return {
            "baseline_player1": self.baselines["player1"],
            "baseline_player2": self.baselines["player2"],
            "updates_player1": self.update_counts["player1"],
            "updates_player2": self.update_counts["player2"],
        }


def play_strategy_game(model, tokenizer, device, game_env) -> StrategyGameTrajectory:
    """
    Play a single strategy game between two instances of the same model.

    Returns complete trajectory data for training.
    """
    # Generate strategy code for both players
    player_submissions = []

    for player_id in [0, 1]:
        # Get prompt for this player
        prompt = game_env.get_player_prompt(player_id, "player")

        # Generate response
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        if device.type == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config["generation"]["max_new_tokens"],
                temperature=config["generation"]["temperature"],
                top_p=config["generation"]["top_p"],
                top_k=(
                    config["generation"]["top_k"]
                    if config["generation"]["top_k"] > 0
                    else None
                ),
                do_sample=config["generation"]["do_sample"],
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        # Extract and validate code
        extracted_code = game_env.extract_code_from_response(response)
        is_valid, error_msg = game_env.validate_submission(
            extracted_code, player_id, "player"
        )

        # Create player submission
        submission = PlayerSubmission(
            player_id=player_id,
            role="player",
            prompt=prompt,
            response=response,
            extracted_code=extracted_code,
            compilation_success=is_valid,
            compilation_error=error_msg,
        )
        player_submissions.append(submission)

    # Play the game
    game_result = game_env.play_game(player_submissions)

    return StrategyGameTrajectory(
        player1_data={
            "prompt": player_submissions[0].prompt,
            "response": player_submissions[0].response,
            "code": player_submissions[0].extracted_code,
            "compilation_success": player_submissions[0].compilation_success,
            "compilation_error": player_submissions[0].compilation_error,
            "role": "player1",
        },
        player2_data={
            "prompt": player_submissions[1].prompt,
            "response": player_submissions[1].response,
            "code": player_submissions[1].extracted_code,
            "compilation_success": player_submissions[1].compilation_success,
            "compilation_error": player_submissions[1].compilation_error,
            "role": "player2",
        },
        game_outcome={
            "player1_reward": game_result.player_rewards.get(0, -1.0),
            "player2_reward": game_result.player_rewards.get(1, -1.0),
            "successful_submissions": game_result.successful_submissions,
        },
        game_result=game_result,
    )


def compute_policy_gradient_loss(
    model,
    tokenizer,
    trajectories: List[StrategyGameTrajectory],
    rae: RoleConditionedAdvantageEstimation,
    device,
) -> torch.Tensor:
    """
    Compute policy gradient loss using SPIRAL's RAE method with memory optimization.
    """
    total_loss = torch.zeros(1, device=device, requires_grad=True)
    num_updates = 0

    for trajectory in trajectories:
        # Process both players
        for role in ["player1", "player2"]:
            if role == "player1":
                player_data = trajectory.player1_data
                reward = trajectory.game_outcome["player1_reward"]
            else:
                player_data = trajectory.player2_data
                reward = trajectory.game_outcome["player2_reward"]

            # Update RAE baseline for this role
            rae.update_baseline(role, reward)

            # Compute advantage using RAE
            advantage = rae.compute_advantage(role, reward)
            advantage_tensor = torch.tensor(
                advantage, device=device, dtype=torch.float32
            ).detach()

            # Process this player's trajectory
            prompt = player_data["prompt"]
            response = player_data["response"]

            # Tokenize with shorter max lengths to save memory
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            )
            response_tokens = tokenizer(
                response, return_tensors="pt", truncation=True, max_length=256
            )

            if device.type == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
                response_tokens = {k: v.to(device) for k, v in response_tokens.items()}

            # Skip if response is too short
            if response_tokens["input_ids"].shape[1] == 0:
                continue

            # Forward pass
            with torch.no_grad():
                # Get full sequence (prompt + response)
                full_input_ids = torch.cat(
                    [inputs["input_ids"], response_tokens["input_ids"]], dim=1
                )
                full_attention_mask = torch.ones_like(full_input_ids)

            # Ensure correct dtype
            full_input_ids = full_input_ids.long()
            full_attention_mask = full_attention_mask.long()

            # Forward pass for loss computation
            outputs = model(
                input_ids=full_input_ids, attention_mask=full_attention_mask
            )
            logits = outputs.logits

            # Compute log probabilities for response tokens only
            response_start_idx = inputs["input_ids"].shape[1]
            response_logits = logits[
                :, response_start_idx - 1 : -1, :
            ]  # Shift for next-token prediction
            response_targets = response_tokens["input_ids"][:, :]

            # Ensure dimensions match
            min_len = min(response_logits.shape[1], response_targets.shape[1])
            response_logits = response_logits[:, :min_len, :]
            response_targets = response_targets[:, :min_len]

            # Compute log probabilities
            log_probs = F.log_softmax(response_logits, dim=-1)
            token_log_probs = log_probs.gather(
                2, response_targets.unsqueeze(-1)
            ).squeeze(-1)

            # REINFORCE loss: -advantage * sum(log_probs)
            sequence_log_prob = token_log_probs.sum()
            loss = -advantage_tensor * sequence_log_prob

            total_loss = total_loss + loss
            num_updates += 1

            # Clear intermediate tensors to free memory
            del outputs, logits, response_logits, log_probs, token_log_probs
            torch.cuda.empty_cache()

    return total_loss / max(num_updates, 1)


# Set up model cache directory
cache_dir = config["model"]["cache_dir"]
os.makedirs(cache_dir, exist_ok=True)

# Use exactly the model specified in config
model_id = config["model"]["id"]
print(f"📥 Using model from config: {model_id}")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_dir,
    local_files_only=offline_mode,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id, cache_dir=cache_dir, local_files_only=offline_mode
)

# Ensure tokenizer has pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Add LoRA for efficient training
print("🔧 Setting up LoRA configuration...")
lora_config = LoraConfig(
    task_type=config["lora"]["task_type"],
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["lora_alpha"],
    target_modules=config["lora"]["target_modules"],
)
print("🎯 Applying LoRA to model...")
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())

# Get device
device = next(model.parameters()).device
print(f"Model device: {device}")

# Initialize game environment
game_env = IteratedPrisonersDilemma(config["game"])
print(
    f"🎮 Initialized {game_env.get_game_name()} with {config['game']['num_rounds']} rounds"
)

# Initialize SPIRAL components
rae = RoleConditionedAdvantageEstimation(alpha=config["training"]["rae_alpha"])
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

# Training parameters
num_steps = config["training"]["num_steps"]

# Adaptive batch size based on GPU memory
if platform_info["gpu_type"] == "V100":
    games_per_step = config["training"]["games_per_step_v100"]
    print(f"🔧 Using V100 memory-optimized settings: {games_per_step} games per step")
else:
    games_per_step = config["training"]["games_per_step_other"]
    print(f"🔧 Using standard memory settings: {games_per_step} games per step")

print(
    f"🎮 Starting SPIRAL prisoner's dilemma training: {num_steps} steps, {games_per_step} games per step"
)

# Initialize wandb run (only if W&B is enabled by user)
if WANDB_ENABLED:
    # Create human-readable timestamp: Jul31_2025_14h30m
    timestamp = datetime.datetime.now().strftime("%b%d_%Y_%Hh%Mm")
    project_name = f"{config['wandb']['project_name_prefix']}-{timestamp}"
    wandb.init(project=project_name)
    print(
        f"✅ Initialized W&B run: {wandb.run.name} (Project: {project_name}, Offline mode: {offline_mode})"
    )

# Run initial MBPP evaluation if enabled
if config["evaluation"].get("enabled_initial", True) and mbpp_evaluator.config.enabled:
    print("🧪 Running initial MBPP evaluation...")
    initial_results = mbpp_evaluator.evaluate_model(
        model, tokenizer, step=0, phase="initial"
    )

    if WANDB_ENABLED and wandb.run and "pass_rate" in initial_results:
        wandb.log(
            {
                "mbpp_eval/pass_rate": initial_results["pass_rate"],
                "mbpp_eval/problems_passed": initial_results["problems_passed"],
                "mbpp_eval/total_problems": initial_results["total_problems"],
                "mbpp_eval/eval_time": initial_results["eval_time_seconds"],
                "step": 0,
            }
        )

# Training loop
for step in range(num_steps):
    print(f"\n🎯 Step {step + 1}/{num_steps}")

    # Collect trajectories through self-play games
    trajectories = []
    player1_wins = 0
    player2_wins = 0
    ties = 0
    successful_games = 0
    execution_failures = 0

    for game_idx in range(games_per_step):
        trajectory = play_strategy_game(model, tokenizer, device, game_env)
        trajectories.append(trajectory)

        # Track statistics
        p1_reward = trajectory.game_outcome["player1_reward"]
        p2_reward = trajectory.game_outcome["player2_reward"]

        if p1_reward > p2_reward:
            player1_wins += 1
        elif p2_reward > p1_reward:
            player2_wins += 1
        else:
            ties += 1

        if trajectory.game_outcome["successful_submissions"] == 2:
            successful_games += 1
        else:
            execution_failures += 1

        # Show detailed results for first few games
        if game_idx < 2:  # Show first 2 games
            print(f"\n{'='*60}")
            print(f"🎮 Game {game_idx + 1}/{games_per_step} - Step {step + 1}")
            print(f"{'='*60}")

            # Show player strategies (truncated)
            print("🤖 Player 1 Strategy:")
            print(f"   Code: {trajectory.player1_data['code'][:100]}...")
            print(
                f"   Compilation: {'✅' if trajectory.player1_data['compilation_success'] else '❌'}"
            )
            if not trajectory.player1_data["compilation_success"]:
                print(f"   Error: {trajectory.player1_data['compilation_error']}")

            print("🤖 Player 2 Strategy:")
            print(f"   Code: {trajectory.player2_data['code'][:100]}...")
            print(
                f"   Compilation: {'✅' if trajectory.player2_data['compilation_success'] else '❌'}"
            )
            if not trajectory.player2_data["compilation_success"]:
                print(f"   Error: {trajectory.player2_data['compilation_error']}")

            # Show game results
            if (
                hasattr(trajectory.game_result, "game_data")
                and "final_payoffs" in trajectory.game_result.game_data
            ):
                payoffs = trajectory.game_result.game_data["final_payoffs"]
                print(f"\n🏆 Final Scores:")
                print(f"   Player 1: {payoffs.get(0, 'N/A')} points")
                print(f"   Player 2: {payoffs.get(1, 'N/A')} points")
                print(
                    f"   Winner: {trajectory.game_result.game_data.get('winner', 'N/A')}"
                )

            print(f"\n🎯 Rewards:")
            print(f"   Player 1: {p1_reward:+.1f}")
            print(f"   Player 2: {p2_reward:+.1f}")
            print(f"{'='*60}")

    # Compute policy gradient loss using RAE
    loss = compute_policy_gradient_loss(model, tokenizer, trajectories, rae, device)

    # Update model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=config["training"]["gradient_clip_norm"]
    )
    optimizer.step()

    # Logging
    rae_stats = rae.get_stats()
    success_rate = successful_games / games_per_step

    stats = {
        "step": step,
        "loss": loss.item(),
        "player1_wins": player1_wins,
        "player2_wins": player2_wins,
        "ties": ties,
        "success_rate": success_rate,
        "execution_failures": execution_failures,
        **rae_stats,
    }

    print(f"📊 Loss: {loss.item():.4f}")
    print(
        f"🏆 Player 1 wins: {player1_wins}, Player 2 wins: {player2_wins}, Ties: {ties}"
    )
    print(f"💻 Success rate: {success_rate:.2%}")
    print(f"❌ Execution failures: {execution_failures}")
    print(
        f"📈 Baselines - P1: {rae_stats['baseline_player1']:.3f}, P2: {rae_stats['baseline_player2']:.3f}"
    )

    if WANDB_ENABLED and wandb.run:
        wandb.log(stats)

    # Run interval MBPP evaluation if enabled
    if (
        config["evaluation"].get("enabled_interval", False)
        and mbpp_evaluator.config.enabled
        and (step + 1) % config["evaluation"]["eval_interval_steps"] == 0
    ):
        print(f"🧪 Running interval MBPP evaluation at step {step + 1}...")
        interval_results = mbpp_evaluator.evaluate_model(
            model, tokenizer, step=step + 1, phase="interval"
        )

        if WANDB_ENABLED and wandb.run and "pass_rate" in interval_results:
            wandb.log(
                {
                    "mbpp_eval/pass_rate": interval_results["pass_rate"],
                    "mbpp_eval/problems_passed": interval_results["problems_passed"],
                    "mbpp_eval/total_problems": interval_results["total_problems"],
                    "mbpp_eval/eval_time": interval_results["eval_time_seconds"],
                    "step": step + 1,
                }
            )

    # Save checkpoint periodically
    if (step + 1) % config["training"]["save_interval"] == 0:
        checkpoint_dir = f"{config['training']['checkpoint_dir']}/step_{step + 1}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Save RAE state
        with open(f"{checkpoint_dir}/rae_state.json", "w") as f:
            json.dump(rae_stats, f, indent=2)

        print(f"💾 Saved checkpoint at step {step + 1}")

print("🏁 SPIRAL prisoner's dilemma training completed!")

# Run final MBPP evaluation if enabled
if config["evaluation"].get("enabled_final", True) and mbpp_evaluator.config.enabled:
    print("🧪 Running final MBPP evaluation...")
    final_results = mbpp_evaluator.evaluate_model(
        model, tokenizer, step=num_steps, phase="final"
    )

    if WANDB_ENABLED and wandb.run and "pass_rate" in final_results:
        wandb.log(
            {
                "mbpp_eval/pass_rate": final_results["pass_rate"],
                "mbpp_eval/problems_passed": final_results["problems_passed"],
                "mbpp_eval/total_problems": final_results["total_problems"],
                "mbpp_eval/eval_time": final_results["eval_time_seconds"],
                "step": num_steps,
            }
        )

# Final checkpoint
final_checkpoint_dir = f"{config['training']['checkpoint_dir']}/final"
os.makedirs(final_checkpoint_dir, exist_ok=True)
model.save_pretrained(final_checkpoint_dir)
tokenizer.save_pretrained(final_checkpoint_dir)

with open(f"{final_checkpoint_dir}/rae_state.json", "w") as f:
    json.dump(rae.get_stats(), f, indent=2)

print(f"💾 Saved final checkpoint")

if WANDB_ENABLED and wandb.run:
    wandb.finish()
    print("✅ Training completed (W&B run finished)")
else:
    print("✅ Training completed (W&B logging was disabled)")
