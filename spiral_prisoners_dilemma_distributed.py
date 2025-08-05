#!/usr/bin/env python3
"""
Distributed SPIRAL Iterated Prisoner's Dilemma Training

Multi-GPU self-play training where LLMs compete by submitting code strategies for the
iterated prisoner's dilemma. Uses PyTorch DDP for distributed training across multiple GPUs.

Key Features:
- Distributed game generation across multiple GPUs
- Synchronized RAE baselines across processes
- LLMs submit code strategies, not direct actions
- Reward based on tournament wins/losses
- -1 reward for code execution failures
- Extensible game framework for other strategy games
"""

# Set environment variable to avoid tokenizers warning
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable CUDA debugging if requested - this helps debug the "device-side assert" errors
if os.environ.get("CUDA_DEBUG", "false").lower() == "true":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    print(
        "🔍 CUDA debugging enabled - blocking execution and device-side assertions activated"
    )

# Set longer NCCL timeout to prevent distributed training timeouts
os.environ["NCCL_TIMEOUT"] = "3600"  # 1 hour instead of default 10 minutes
os.environ["NCCL_BLOCKING_WAIT"] = "1"  # More reliable error reporting
print("⏰ Set NCCL timeout to 1 hour to prevent distributed sync timeouts")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
import concurrent.futures
import multiprocessing
import functools
import time


# Initialize distributed training first
def init_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))

        print(f"[Rank {rank}] Initializing distributed SPIRAL training...")
        print(f"[Rank {rank}] World size: {world_size}, Local rank: {local_rank}")

        # Initialize the process group
        dist.init_process_group(backend="nccl")

        # Set the device for this process
        torch.cuda.set_device(local_rank)

        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "is_main_process": rank == 0,
        }
    else:
        print("⚠️ Distributed environment not detected, running in single-GPU mode")
        return {"rank": 0, "world_size": 1, "local_rank": 0, "is_main_process": True}


# Initialize distributed training
dist_info = init_distributed()

# Only print startup info on main process
if dist_info["is_main_process"]:
    print("🚀 Starting Distributed SPIRAL Prisoner's Dilemma Training...")
    print("📋 Initializing components...")

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Distributed SPIRAL Prisoner's Dilemma Training"
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/spiral_prisoners_dilemma.yaml",
    help="Path to configuration file",
)

# Parse known args to allow config overrides
args, unknown_args = parser.parse_known_args()

# Load configuration
if dist_info["is_main_process"]:
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

        # Validate that the config path exists (except for the final key)
        for i, k in enumerate(keys[:-1]):
            if k not in current:
                if dist_info["is_main_process"]:
                    print(f"❌ ERROR: Invalid config override path '{key}'")
                    print(f"   Key '{k}' does not exist in config at level {i+1}")
                    print(f"   Available keys at this level: {list(current.keys())}")
                raise ValueError(f"Invalid config override path: {key}")
            current = current[k]

        old_value = current.get(keys[-1], "NOT_SET")
        current[keys[-1]] = value
        overrides_applied.append(f"{key}: {old_value} -> {value}")

        i += 1

    if overrides_applied and dist_info["is_main_process"]:
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
    if dist_info["is_main_process"]:
        print("🔍 Detecting platform and GPU capabilities...")
    if not torch.cuda.is_available():
        return {
            "supports_bf16": False,
            "device": "cpu",
            "gpu_type": "none",
            "offline_mode": False,
            "platform": "cpu",
            "offline_reason": "no GPU available",
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
if dist_info["is_main_process"]:
    print("🔧 Running platform detection...")
platform_info = detect_platform_and_gpu()
if dist_info["is_main_process"]:
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
        if dist_info["is_main_process"]:
            print("✅ Set global offline mode for transformers and wandb")
    else:
        # For online mode, ensure offline flags are not set
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        os.environ.pop("WANDB_MODE", None)
else:  # If WANDB_ENABLED is False, explicitly disable W&B
    os.environ["WANDB_MODE"] = "disabled"
    if dist_info["is_main_process"]:
        print("🚫 W&B logging explicitly disabled by user configuration.")

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if dist_info["is_main_process"]:
    print("📦 Loading utility modules...")
from utils.env_loader import get_api_key
from utils.seed_manager import SeedManager
from evaluation.mbpp.evaluator import MBPPEvaluator, EvalConfig
from game_environments.prisoners_dilemma import IteratedPrisonersDilemma
from game_environments.base_game import PlayerSubmission

# Initialize wandb with API key from environment (skip if W&B is not enabled)
if WANDB_ENABLED and dist_info["is_main_process"]:  # Only main process initializes W&B
    wandb_key = get_api_key("wandb", required=False)
    if wandb_key:
        wandb.login(key=wandb_key)
        print("✓ Logged into W&B using environment variable")
    else:
        print(
            "⚠️ No W&B API key found, continuing with W&B logging (if online) or local saves (if offline)."
        )
elif dist_info["is_main_process"]:
    print("🚫 Skipping W&B login (W&B is disabled by user).")

# Initialize comprehensive seed management
if dist_info["is_main_process"]:
    print("🎲 Setting up seed management...")
seed_manager = SeedManager.from_config(config)
seed_manager.seed_everything()

# Initialize MBPP evaluator with consolidated config (only on main process)
if dist_info["is_main_process"]:
    print("🧪 Setting up MBPP evaluator...")

    # Create evaluation config from main config
    eval_config_dict = config.get("evaluation", {}).copy()

    # Remove keys not expected by EvalConfig constructor
    eval_config_dict.pop("enabled_initial", None)
    eval_config_dict.pop("enabled_final", None)
    eval_config_dict.pop("enabled_interval", None)
    eval_config_dict.pop("eval_interval_steps", None)
    eval_config_dict.pop("consistent_questions", None)

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
else:
    mbpp_evaluator = None


@dataclass
class StrategyGameTrajectory:
    """Store trajectory data for a single strategy game."""

    player1_data: Dict  # Player 1's submission and trajectory
    player2_data: Dict  # Player 2's submission and trajectory
    game_outcome: Dict  # Final results and rewards
    game_result: Any  # Full game result from the environment


class DistributedRoleConditionedAdvantageEstimation:
    """
    Distributed Role-conditioned Advantage Estimation (RAE) from SPIRAL paper.
    Maintains separate baselines for each player role to reduce variance.
    Synchronizes baselines across distributed processes.
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

    def sync_baselines(self):
        """Synchronize baselines across all distributed processes with timeout protection."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            try:
                if dist_info["is_main_process"]:
                    print(
                        f"🔄 [Rank {dist_info['rank']}] Syncing RAE baselines across {dist.get_world_size()} processes..."
                    )

                # Convert baselines to tensors for all_reduce
                baseline_tensor = torch.tensor(
                    [self.baselines["player1"], self.baselines["player2"]],
                    dtype=torch.float32,
                    device=f"cuda:{dist_info['local_rank']}",
                )

                count_tensor = torch.tensor(
                    [self.update_counts["player1"], self.update_counts["player2"]],
                    dtype=torch.float32,
                    device=f"cuda:{dist_info['local_rank']}",
                )

                # Barrier to ensure all processes are ready
                if dist_info["is_main_process"]:
                    print(
                        f"🔄 [Rank {dist_info['rank']}] Waiting for all processes at RAE barrier..."
                    )
                dist.barrier(timeout=datetime.timedelta(seconds=30))

                # All-reduce operations
                if dist_info["is_main_process"]:
                    print(
                        f"🔄 [Rank {dist_info['rank']}] Performing RAE all_reduce operations..."
                    )

                dist.all_reduce(baseline_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

                if dist_info["is_main_process"]:
                    print(
                        f"✅ [Rank {dist_info['rank']}] RAE all_reduce completed successfully"
                    )

                # Average the baselines
                world_size = dist.get_world_size()
                baseline_tensor /= world_size
                count_tensor /= world_size

                # Update local baselines
                self.baselines["player1"] = baseline_tensor[0].item()
                self.baselines["player2"] = baseline_tensor[1].item()
                self.update_counts["player1"] = int(count_tensor[0].item())
                self.update_counts["player2"] = int(count_tensor[1].item())

                if dist_info["is_main_process"]:
                    print(
                        f"📊 [Rank {dist_info['rank']}] RAE sync complete - P1: {self.baselines['player1']:.3f}, P2: {self.baselines['player2']:.3f}"
                    )

            except Exception as e:
                if dist_info["is_main_process"]:
                    print(f"⚠️ [Rank {dist_info['rank']}] RAE baseline sync failed: {e}")
                    print(f"   Continuing with local baselines only")
                # Continue with local baselines if sync fails

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


def execute_game_parallel(game_data_with_index):
    """
    Helper function for parallel game execution.
    This function is designed to run in a separate process for CPU parallelism.

    Args:
        game_data_with_index: Tuple of (index, game_submissions, game_env_config)

    Returns:
        Tuple of (index, game_result, execution_success)
    """
    try:
        idx, player_submissions, game_env_config = game_data_with_index

        # Recreate game environment in this process (needed for multiprocessing)
        from game_environments.prisoners_dilemma import IteratedPrisonersDilemma

        local_game_env = IteratedPrisonersDilemma(game_env_config)

        # Execute the game
        game_result = local_game_env.play_game(player_submissions)

        return idx, game_result, True

    except Exception as e:
        # Return error information
        return (
            idx,
            {
                "error": str(e),
                "player_rewards": {0: -1.0, 1: -1.0},
                "successful_submissions": 0,
            },
            False,
        )


def play_strategy_game_distributed(
    model, tokenizer, device, game_env, step: int = 0, rank: int = 0
) -> StrategyGameTrajectory:
    """
    Play a single strategy game between two instances of the same model.
    Distributed version that includes rank in seeding for reproducibility.

    Returns complete trajectory data for training.
    """
    # Generate strategy code for both players
    player_submissions = []

    for player_id in [0, 1]:
        # Get prompt for this player
        prompt = game_env.get_player_prompt(player_id, "player")

        # Generate response with seeded randomness (include rank for distributed seeding)
        try:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )

            # Validate input IDs are within vocabulary bounds
            vocab_size = tokenizer.vocab_size
            if torch.any(inputs["input_ids"] >= vocab_size) or torch.any(
                inputs["input_ids"] < 0
            ):
                raise ValueError(
                    f"Input token IDs out of bounds. Max ID: {torch.max(inputs['input_ids'])}, Vocab size: {vocab_size}"
                )

            if device.type == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Seed for deterministic generation per player (include rank for distributed consistency)
            seed_manager.seed_for_generation(
                step=step, generation_idx=player_id + rank * 2
            )

            with torch.no_grad():
                # Use .module to access the underlying model when using DDP
                generation_model = model.module if hasattr(model, "module") else model

                # Additional safety checks
                if inputs["input_ids"].shape[1] > 1024:
                    if dist_info["is_main_process"]:
                        print(
                            f"⚠️ Warning: Input length {inputs['input_ids'].shape[1]} exceeds 1024, truncating..."
                        )
                    inputs["input_ids"] = inputs["input_ids"][:, :1024]
                    inputs["attention_mask"] = inputs["attention_mask"][:, :1024]

                outputs = generation_model.generate(
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
                    # Additional safety parameters
                    max_length=inputs["input_ids"].shape[1]
                    + config["generation"]["max_new_tokens"],
                    use_cache=True,
                )
        except Exception as e:
            if dist_info["is_main_process"]:
                print(f"❌ Generation error for Player {player_id} in game {step}: {e}")
                print(f"   Prompt length: {len(prompt)}")
                print(
                    f"   Input shape: {inputs.get('input_ids', torch.tensor([])).shape if 'inputs' in locals() else 'N/A'}"
                )
            # Return a fallback response
            outputs = inputs["input_ids"]  # Just return the input as fallback

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


def compute_distributed_policy_gradient_loss(
    model,
    tokenizer,
    trajectories: List[StrategyGameTrajectory],
    rae: DistributedRoleConditionedAdvantageEstimation,
    device,
) -> torch.Tensor:
    """
    Compute distributed policy gradient loss using SPIRAL's RAE method.
    Each process computes loss for its subset of trajectories.
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
if dist_info["is_main_process"]:
    print(f"📥 Using model from config: {model_id}")

# Load model and tokenizer
if dist_info["is_main_process"]:
    print(f"📥 Loading model: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map=f"cuda:{dist_info['local_rank']}",  # Explicit device placement
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
if dist_info["is_main_process"]:
    print("🔧 Setting up LoRA configuration...")
lora_config = LoraConfig(
    task_type=config["lora"]["task_type"],
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["lora_alpha"],
    target_modules=config["lora"]["target_modules"],
)
if dist_info["is_main_process"]:
    print("🎯 Applying LoRA to model...")
model = get_peft_model(model, lora_config)
if dist_info["is_main_process"]:
    print(model.print_trainable_parameters())

# Wrap model with DDP
if dist_info["world_size"] > 1:
    if dist_info["is_main_process"]:
        print("🌐 Wrapping model with DistributedDataParallel...")
    model = DDP(
        model,
        device_ids=[dist_info["local_rank"]],
        output_device=dist_info["local_rank"],
    )

# Get device
device = torch.device(f"cuda:{dist_info['local_rank']}")
if dist_info["is_main_process"]:
    print(f"Model device: {device}")

# Initialize game environment
game_env = IteratedPrisonersDilemma(config["game"])
if dist_info["is_main_process"]:
    print(
        f"🎮 Initialized {game_env.get_game_name()} with {config['game']['num_rounds']} rounds"
    )

# Initialize SPIRAL components
rae = DistributedRoleConditionedAdvantageEstimation(
    alpha=config["training"]["rae_alpha"]
)
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

# Training parameters
num_steps = config["training"]["num_steps"]

# Adaptive batch size based on GPU memory and distributed setup
if platform_info["gpu_type"] == "V100":
    games_per_step_per_gpu = config["training"]["games_per_step_v100"]
    if dist_info["is_main_process"]:
        print(
            f"🔧 Using V100 memory-optimized settings: {games_per_step_per_gpu} games per GPU"
        )
else:
    games_per_step_per_gpu = config["training"]["games_per_step_other"]
    if dist_info["is_main_process"]:
        print(
            f"🔧 Using standard memory settings: {games_per_step_per_gpu} games per GPU"
        )

total_games_per_step = games_per_step_per_gpu * dist_info["world_size"]

if dist_info["is_main_process"]:
    print(f"🎮 Starting distributed SPIRAL prisoner's dilemma training:")
    print(
        f"   {num_steps} steps, {games_per_step_per_gpu} games per GPU, {total_games_per_step} games total per step"
    )
    print(f"   Using {dist_info['world_size']} GPUs")

# Initialize wandb run (only if W&B is enabled by user and on main process)
if WANDB_ENABLED and dist_info["is_main_process"]:
    # Create human-readable timestamp: Jul31_2025_14h30m
    timestamp = datetime.datetime.now().strftime("%b%d_%Y_%Hh%Mm")
    project_name = f"{config['wandb']['project_name_prefix']}-distributed-{timestamp}"
    wandb.init(
        project=project_name,
        config={**config, **seed_manager.get_seed_info(), "distributed": dist_info},
    )
    print(
        f"✅ Initialized W&B run: {wandb.run.name} (Project: {project_name}, Offline mode: {offline_mode})"
    )

# Run initial MBPP evaluation if enabled (only on main process)
if (
    config["evaluation"].get("enabled_initial", True)
    and mbpp_evaluator is not None
    and mbpp_evaluator.config.enabled
    and dist_info["is_main_process"]
):
    print("🧪 Running initial MBPP evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("initial")
    # Use the underlying model for evaluation (unwrap DDP if needed)
    eval_model = model.module if hasattr(model, "module") else model
    initial_results = mbpp_evaluator.evaluate_model(
        eval_model, tokenizer, step=0, phase="initial"
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
    if dist_info["is_main_process"]:
        print(f"\n🎯 Step {step + 1}/{num_steps}")

    # Collect trajectories through distributed self-play games
    trajectories = []
    local_stats = {
        "player1_wins": 0,
        "player2_wins": 0,
        "ties": 0,
        "successful_games": 0,
        "execution_failures": 0,
        "total_noise_rounds": 0,
        "games_with_noise": 0,
        "wsls_bot_games": 0,
        "wsls_bot_wins": 0,
    }

    # ============================================================================
    # PHASE 1: Generate strategies for all games (GPU-bound, sequential)
    # ============================================================================
    if dist_info["is_main_process"]:
        print(
            f"🔥 [Rank {dist_info['rank']}] Phase 1: Generating strategies for {games_per_step_per_gpu} games..."
        )

    game_preparations = []  # Will store (game_idx, player_submissions) for each game

    for game_idx in range(games_per_step_per_gpu):
        # Calculate global game index for seeding consistency
        global_game_idx = dist_info["rank"] * games_per_step_per_gpu + game_idx
        game_step = step * total_games_per_step + global_game_idx

        # Generate strategy code for both players
        player_submissions = []

        for player_id in [0, 1]:
            # Get prompt for this player
            prompt = game_env.get_player_prompt(player_id, "player")

            # Generate response with seeded randomness (include rank for distributed consistency)
            try:
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=1024
                )

                # Validate input IDs are within vocabulary bounds
                vocab_size = tokenizer.vocab_size
                if torch.any(inputs["input_ids"] >= vocab_size) or torch.any(
                    inputs["input_ids"] < 0
                ):
                    raise ValueError(
                        f"Input token IDs out of bounds. Max ID: {torch.max(inputs['input_ids'])}, Vocab size: {vocab_size}"
                    )

                if device.type == "cuda":
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                # Seed for deterministic generation per player (include rank for distributed consistency)
                seed_manager.seed_for_generation(
                    step=game_step, generation_idx=player_id + dist_info["rank"] * 2
                )

                with torch.no_grad():
                    # Use .module to access the underlying model when using DDP
                    generation_model = (
                        model.module if hasattr(model, "module") else model
                    )

                    # Additional safety checks
                    if inputs["input_ids"].shape[1] > 1024:
                        if dist_info["is_main_process"]:
                            print(
                                f"⚠️ Warning: Input length {inputs['input_ids'].shape[1]} exceeds 1024, truncating..."
                            )
                        inputs["input_ids"] = inputs["input_ids"][:, :1024]
                        inputs["attention_mask"] = inputs["attention_mask"][:, :1024]

                    outputs = generation_model.generate(
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
                        # Additional safety parameters
                        max_length=inputs["input_ids"].shape[1]
                        + config["generation"]["max_new_tokens"],
                        use_cache=True,
                    )
            except Exception as e:
                if dist_info["is_main_process"]:
                    print(
                        f"❌ Generation error for Player {player_id} in game {game_idx}, step {step}: {e}"
                    )
                    print(f"   Prompt length: {len(prompt)}")
                    print(
                        f"   Input shape: {inputs.get('input_ids', torch.tensor([])).shape if 'inputs' in locals() else 'N/A'}"
                    )
                # Return a fallback response
                outputs = inputs["input_ids"]  # Just return the input as fallback

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

        # Store game preparation data for parallel execution
        game_preparations.append((game_idx, player_submissions))

    # ============================================================================
    # PHASE 2: Execute all games in parallel (CPU-bound, parallel)
    # ============================================================================
    if dist_info["is_main_process"]:
        print(
            f"⚙️ [Rank {dist_info['rank']}] Phase 2: Executing {len(game_preparations)} games in parallel..."
        )

    # Determine number of workers for game execution
    max_workers = min(len(game_preparations), multiprocessing.cpu_count())

    # Prepare data for parallel execution
    game_execution_data = [
        (game_idx, player_submissions, config["game"])
        for game_idx, player_submissions in game_preparations
    ]

    # Execute games in parallel
    game_results = {}
    parallel_start = time.time()

    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Submit all game execution tasks
            future_to_index = {
                executor.submit(execute_game_parallel, game_data): game_data[0]
                for game_data in game_execution_data
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    idx, game_result, success = future.result()
                    game_results[idx] = (game_result, success)
                except Exception as e:
                    idx = future_to_index[future]
                    game_results[idx] = (
                        {
                            "error": str(e),
                            "player_rewards": {0: -1.0, 1: -1.0},
                            "successful_submissions": 0,
                        },
                        False,
                    )
    except Exception as e:
        # Fallback to sequential execution if parallel fails
        if dist_info["is_main_process"]:
            print(
                f"⚠️ [Rank {dist_info['rank']}] Parallel game execution failed: {e}, falling back to sequential"
            )

        for game_idx, player_submissions in game_preparations:
            try:
                game_result = game_env.play_game(player_submissions)
                game_results[game_idx] = (game_result, True)
            except Exception as e:
                game_results[game_idx] = (
                    {
                        "error": str(e),
                        "player_rewards": {0: -1.0, 1: -1.0},
                        "successful_submissions": 0,
                    },
                    False,
                )

    parallel_time = time.time() - parallel_start

    if dist_info["is_main_process"]:
        print(
            f"🚀 [Rank {dist_info['rank']}] Game execution completed in {parallel_time:.2f}s using {max_workers} workers"
        )

    # ============================================================================
    # PHASE 3: Process results and create trajectories
    # ============================================================================
    for game_idx, player_submissions in game_preparations:
        game_result, execution_success = game_results[game_idx]

        # Create trajectory object
        trajectory = StrategyGameTrajectory(
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
                "player1_reward": (
                    game_result.player_rewards.get(0, -1.0)
                    if hasattr(game_result, "player_rewards")
                    else game_result.get("player_rewards", {}).get(0, -1.0)
                ),
                "player2_reward": (
                    game_result.player_rewards.get(1, -1.0)
                    if hasattr(game_result, "player_rewards")
                    else game_result.get("player_rewards", {}).get(1, -1.0)
                ),
                "successful_submissions": (
                    game_result.successful_submissions
                    if hasattr(game_result, "successful_submissions")
                    else game_result.get("successful_submissions", 0)
                ),
            },
            game_result=game_result,
        )
        trajectories.append(trajectory)

        # Track local statistics
        p1_reward = trajectory.game_outcome["player1_reward"]
        p2_reward = trajectory.game_outcome["player2_reward"]

        if p1_reward > p2_reward:
            local_stats["player1_wins"] += 1
        elif p2_reward > p1_reward:
            local_stats["player2_wins"] += 1
        else:
            local_stats["ties"] += 1

        if trajectory.game_outcome["successful_submissions"] == 2:
            local_stats["successful_games"] += 1
        else:
            local_stats["execution_failures"] += 1

        # Track noise and WSLS bot statistics (only if game executed successfully)
        if execution_success and hasattr(game_result, "game_data"):
            game_data = game_result.game_data
            if "noise_rounds" in game_data:
                noise_rounds = game_data["noise_rounds"]
                local_stats["total_noise_rounds"] += noise_rounds
                if noise_rounds > 0:
                    local_stats["games_with_noise"] += 1

            if game_data.get("wsls_bot_used", False):
                local_stats["wsls_bot_games"] += 1
                # Check if WSLS bot (player 2) won
                if p2_reward > p1_reward:
                    local_stats["wsls_bot_wins"] += 1

        # Show detailed results for first few games (only on main process)
        if (
            dist_info["is_main_process"] and game_idx < 2
        ):  # Show fewer games in distributed mode
            debug_config = config.get("debug", {})
            show_detailed_games = debug_config.get(
                "show_detailed_games", 1
            )  # Reduced for distributed
            max_code_chars = debug_config.get("max_code_chars", 200)  # Reduced output
            show_execution_details = debug_config.get(
                "show_execution_details", False
            )  # Reduced verbosity

            if game_idx < show_detailed_games:
                print(f"\n{'='*40}")
                print(
                    f"🎮 [Rank {dist_info['rank']}] Game {game_idx + 1}/{games_per_step_per_gpu} - Step {step + 1}"
                )

                # Show special game features (only if successful execution)
                if execution_success and hasattr(game_result, "game_data"):
                    game_data = game_result.game_data
                    game_features = []
                    if game_data.get("wsls_bot_used", False):
                        game_features.append("🤖 WSLS Bot")
                    if game_data.get("noise_rounds", 0) > 0:
                        game_features.append(
                            f"🎲 Noise ({game_data['noise_rounds']} rounds)"
                        )

                    if game_features:
                        print(f"Features: {' | '.join(game_features)}")

                print(f"🎯 Rewards: P1={p1_reward:+.1f}, P2={p2_reward:+.1f}")
                print(
                    f"⏱️ Parallel execution: {parallel_time:.2f}s with {max_workers} workers"
                )
                print(f"{'='*40}")

    # Compute local policy gradient loss
    local_loss = compute_distributed_policy_gradient_loss(
        model, tokenizer, trajectories, rae, device
    )

    # Synchronize RAE baselines across all processes
    rae.sync_baselines()

    # Average gradients across all processes (DDP handles this automatically)
    optimizer.zero_grad()
    local_loss.backward()

    # Gradient clipping (apply before DDP sync)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=config["training"]["gradient_clip_norm"]
    )

    optimizer.step()

    # Gather statistics from all processes for logging (only on main process)
    if dist_info["is_main_process"]:
        # Convert local stats to tensors for all_reduce with timeout protection
        if dist_info["world_size"] > 1:
            try:
                if dist_info["is_main_process"]:
                    print(
                        f"🔄 [Rank {dist_info['rank']}] Syncing game statistics across {dist_info['world_size']} processes..."
                    )

                stats_tensor = torch.tensor(
                    [
                        local_stats["player1_wins"],
                        local_stats["player2_wins"],
                        local_stats["ties"],
                        local_stats["successful_games"],
                        local_stats["execution_failures"],
                        local_stats["total_noise_rounds"],
                        local_stats["games_with_noise"],
                        local_stats["wsls_bot_games"],
                        local_stats["wsls_bot_wins"],
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                # Barrier to ensure all processes are ready
                if dist_info["is_main_process"]:
                    print(
                        f"🔄 [Rank {dist_info['rank']}] Waiting for all processes at stats barrier..."
                    )
                dist.barrier(timeout=datetime.timedelta(seconds=30))

                # All-reduce operations
                if dist_info["is_main_process"]:
                    print(
                        f"🔄 [Rank {dist_info['rank']}] Performing stats all_reduce..."
                    )

                # Sum across all processes
                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

                if dist_info["is_main_process"]:
                    print(
                        f"✅ [Rank {dist_info['rank']}] Stats all_reduce completed successfully"
                    )

                # Update global stats
                global_stats = {
                    "player1_wins": int(stats_tensor[0].item()),
                    "player2_wins": int(stats_tensor[1].item()),
                    "ties": int(stats_tensor[2].item()),
                    "successful_games": int(stats_tensor[3].item()),
                    "execution_failures": int(stats_tensor[4].item()),
                    "total_noise_rounds": int(stats_tensor[5].item()),
                    "games_with_noise": int(stats_tensor[6].item()),
                    "wsls_bot_games": int(stats_tensor[7].item()),
                    "wsls_bot_wins": int(stats_tensor[8].item()),
                }

                if dist_info["is_main_process"]:
                    print(
                        f"📊 [Rank {dist_info['rank']}] Stats sync complete - {global_stats['successful_games']} successful games"
                    )

            except Exception as e:
                if dist_info["is_main_process"]:
                    print(
                        f"⚠️ [Rank {dist_info['rank']}] Stats synchronization failed: {e}"
                    )
                    print(f"   Using local statistics only")
                # Fallback to local statistics
                global_stats = local_stats
        else:
            global_stats = local_stats

        # Calculate rates and averages
        rae_stats = rae.get_stats()
        success_rate = (
            global_stats["successful_games"] / total_games_per_step
            if total_games_per_step > 0
            else 0
        )
        avg_noise_rounds_per_game = (
            global_stats["total_noise_rounds"] / total_games_per_step
            if total_games_per_step > 0
            else 0
        )
        noise_game_rate = (
            global_stats["games_with_noise"] / total_games_per_step
            if total_games_per_step > 0
            else 0
        )
        wsls_bot_rate = (
            global_stats["wsls_bot_games"] / total_games_per_step
            if total_games_per_step > 0
            else 0
        )
        wsls_bot_win_rate = (
            global_stats["wsls_bot_wins"] / global_stats["wsls_bot_games"]
            if global_stats["wsls_bot_games"] > 0
            else 0
        )

        stats = {
            "step": step,
            "loss": local_loss.item(),
            "total_games_per_step": total_games_per_step,
            "games_per_gpu": games_per_step_per_gpu,
            "world_size": dist_info["world_size"],
            **{f"global_{k}": v for k, v in global_stats.items()},
            "success_rate": success_rate,
            "avg_noise_rounds_per_game": avg_noise_rounds_per_game,
            "noise_game_rate": noise_game_rate,
            "wsls_bot_rate": wsls_bot_rate,
            "wsls_bot_win_rate": wsls_bot_win_rate,
            "parallel_execution_time": parallel_time,
            "cpu_workers": max_workers,
            **rae_stats,
        }

        print(f"📊 Loss: {local_loss.item():.4f}")
        print(
            f"🏆 Global wins - P1: {global_stats['player1_wins']}, P2: {global_stats['player2_wins']}, Ties: {global_stats['ties']}"
        )
        print(
            f"💻 Success rate: {success_rate:.2%} ({global_stats['successful_games']}/{total_games_per_step})"
        )
        print(f"❌ Execution failures: {global_stats['execution_failures']}")
        print(
            f"📈 Baselines - P1: {rae_stats['baseline_player1']:.3f}, P2: {rae_stats['baseline_player2']:.3f}"
        )

        # Print noise and WSLS bot statistics
        if global_stats["total_noise_rounds"] > 0 or global_stats["wsls_bot_games"] > 0:
            print(
                f"🎲 Noise: {avg_noise_rounds_per_game:.1f} rounds/game avg, {noise_game_rate:.1%} games affected"
            )
            if global_stats["wsls_bot_games"] > 0:
                print(
                    f"🤖 WSLS bot: {global_stats['wsls_bot_games']}/{total_games_per_step} games ({wsls_bot_rate:.1%}), won {wsls_bot_win_rate:.1%}"
                )

        # Print parallel execution performance
        print(
            f"⚙️ Game execution: {parallel_time:.2f}s using {max_workers} CPU workers (parallel speedup)"
        )

        if WANDB_ENABLED and wandb.run:
            wandb.log(stats)

    # Synchronize all processes before next step
    if dist_info["world_size"] > 1:
        dist.barrier()

    # Run interval MBPP evaluation if enabled (only on main process)
    if (
        config["evaluation"].get("enabled_interval", False)
        and mbpp_evaluator is not None
        and mbpp_evaluator.config.enabled
        and (step + 1) % config["evaluation"]["eval_interval_steps"] == 0
        and dist_info["is_main_process"]
    ):
        print(f"🧪 Running interval MBPP evaluation at step {step + 1}...")
        # Seed for consistent evaluation
        seed_manager.seed_for_evaluation_auto(f"interval_step_{step + 1}")
        # Use the underlying model for evaluation (unwrap DDP if needed)
        eval_model = model.module if hasattr(model, "module") else model
        interval_results = mbpp_evaluator.evaluate_model(
            eval_model, tokenizer, step=step + 1, phase="interval"
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

    # Save checkpoint periodically (only on main process)
    if (step + 1) % config["training"]["save_interval"] == 0 and dist_info[
        "is_main_process"
    ]:
        checkpoint_dir = f"{config['training']['checkpoint_dir']}/step_{step + 1}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save the underlying model (unwrap DDP if needed)
        save_model = model.module if hasattr(model, "module") else model
        save_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Save RAE state
        with open(f"{checkpoint_dir}/rae_state.json", "w") as f:
            json.dump(rae.get_stats(), f, indent=2)

        print(f"💾 Saved checkpoint at step {step + 1}")

if dist_info["is_main_process"]:
    print("🏁 Distributed SPIRAL prisoner's dilemma training completed!")

# Run final MBPP evaluation if enabled (only on main process)
if (
    config["evaluation"].get("enabled_final", True)
    and mbpp_evaluator is not None
    and mbpp_evaluator.config.enabled
    and dist_info["is_main_process"]
):
    print("🧪 Running final MBPP evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("final")
    # Use the underlying model for evaluation (unwrap DDP if needed)
    eval_model = model.module if hasattr(model, "module") else model
    final_results = mbpp_evaluator.evaluate_model(
        eval_model, tokenizer, step=num_steps, phase="final"
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

# Final checkpoint (only on main process)
if dist_info["is_main_process"]:
    final_checkpoint_dir = f"{config['training']['checkpoint_dir']}/final"
    os.makedirs(final_checkpoint_dir, exist_ok=True)

    # Save the underlying model (unwrap DDP if needed)
    save_model = model.module if hasattr(model, "module") else model
    save_model.save_pretrained(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)

    with open(f"{final_checkpoint_dir}/rae_state.json", "w") as f:
        json.dump(rae.get_stats(), f, indent=2)

    print(f"💾 Saved final checkpoint")

# Clean up distributed
if dist_info["world_size"] > 1:
    dist.destroy_process_group()

if WANDB_ENABLED and wandb.run and dist_info["is_main_process"]:
    wandb.finish()
    print("✅ Distributed training completed (W&B run finished)")
elif dist_info["is_main_process"]:
    print("✅ Distributed training completed (W&B logging was disabled)")
