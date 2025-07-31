#!/usr/bin/env python3
"""
SPIRAL Self-Play Training - Simplified Implementation with MBPP Evaluation
Based on "SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning"

Key SPIRAL concepts implemented:
- Self-play with shared policy
- Role-conditioned advantage estimation (RAE)
- Turn-based zero-sum games
- Multi-turn competitive dynamics
- MBPP evaluation for code generation assessment
"""

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
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
import yaml
import argparse


print("🚀 Starting SPIRAL Self-Play Training with MBPP Evaluation...")
print("📋 Initializing components...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="SPIRAL Self-Play Training")
parser.add_argument(
    "--config",
    type=str,
    default="configs/spiral_self_play.yaml",
    help="Path to configuration file",
)
args = parser.parse_args()

# Load configuration
print(f"📝 Loading configuration from: {args.config}")
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

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
        os.environ["WANDB_MODE"] = "offline"  # Changed from "disabled" to "offline"
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
eval_config_dict = config.get("evaluation", {})

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
    print(f"Initial eval: {'✅' if config['evaluation'].get('enabled_initial', True) else '❌'}")
    print(f"Final eval: {'✅' if config['evaluation'].get('enabled_final', True) else '❌'}")
    print(
        f"Interval eval: {'✅' if config['evaluation'].get('enabled_interval', False) else '❌'}"
    )
    if config["evaluation"].get("enabled_interval", False):
        print(f"Eval every: {config['evaluation'].get('eval_interval_steps', 'N/A')} steps")
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
class GameTrajectory:
    """Store trajectory data for a single game following SPIRAL format."""

    player_trajectories: Dict[int, List[Dict]]  # player_id -> list of moves
    final_rewards: Dict[int, float]  # player_id -> final reward
    game_length: int


class TicTacToeGame:
    """Tic-Tac-Toe implementation following SPIRAL's turn-based zero-sum game format."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to initial state."""
        self.board = [" "] * 9  # 9 positions: 0-8
        self.current_player = 0  # Player 0 (X) starts
        self.game_over = False
        self.winner = None
        self.turn_count = 0

    def get_observation(self, player_id: int) -> str:
        """Get role-conditioned observation following SPIRAL's format."""
        player_symbol = "X" if player_id == 0 else "O"
        opponent_symbol = "O" if player_id == 0 else "X"

        # Format board with current state
        board_str = "Current board:\n"
        for i in range(3):
            row = " | ".join(
                [
                    f"{self.board[i*3 + j] if self.board[i*3 + j] != ' ' else str(i*3 + j)}"
                    for j in range(3)
                ]
            )
            board_str += f" {row} \n"
            if i < 2:
                board_str += "-----------\n"

        # Role-conditioned prompt (key SPIRAL concept)
        prompt = f"""You are playing Tic-Tac-Toe as Player {player_id}. You are {player_symbol}, your opponent is {opponent_symbol}.

{board_str}

Available positions: {self.get_valid_moves()}

Think step by step about your strategy, then choose your move.

<think>
Let me analyze the current board state and find the best move...
</think>

<answer>[position]</answer>"""

        return prompt

    def get_valid_moves(self) -> List[int]:
        """Get list of valid moves (empty positions)."""
        return [i for i in range(9) if self.board[i] == " "]

    def make_move(self, player_id: int, position: int) -> Tuple[bool, bool]:
        """
        Make a move at the given position.
        Returns: (move_successful, game_over)
        """
        if (
            position < 0
            or position > 8
            or self.board[position] != " "
            or self.game_over
        ):
            return False, True  # Invalid move ends game

        symbol = "X" if player_id == 0 else "O"
        self.board[position] = symbol
        self.turn_count += 1

        # Check for winner
        if self.check_winner():
            self.winner = player_id
            self.game_over = True
        elif len(self.get_valid_moves()) == 0:
            # Draw
            self.game_over = True
            self.winner = "Draw"
        else:
            # Switch players
            self.current_player = 1 - self.current_player

        return True, self.game_over

    def check_winner(self) -> bool:
        """Check if current player has won."""
        symbol = "X" if self.current_player == 0 else "O"

        # Winning combinations
        wins = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],  # Rows
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],  # Columns
            [0, 4, 8],
            [2, 4, 6],  # Diagonals
        ]

        for combo in wins:
            if all(self.board[i] == symbol for i in combo):
                return True
        return False

    def get_rewards(self) -> Dict[int, float]:
        """Get zero-sum rewards for both players."""
        if not self.game_over:
            return {0: 0.0, 1: 0.0}

        if self.winner == 0:
            return {0: 1.0, 1: -1.0}
        elif self.winner == 1:
            return {0: -1.0, 1: 1.0}
        else:  # Draw
            return {0: 0.0, 1: 0.0}


class RoleConditionedAdvantageEstimation:
    """
    Role-conditioned Advantage Estimation (RAE) from SPIRAL paper.
    Maintains separate baselines for each player role to reduce variance.
    """

    def __init__(self, alpha: float = 0.95):
        self.alpha = alpha  # EMA decay rate
        self.baselines = {0: 0.0, 1: 0.0}  # Player 0 and Player 1 baselines
        self.update_counts = {0: 0, 1: 0}

    def update_baseline(self, player_id: int, reward: float):
        """Update baseline for specific player role using EMA."""
        if self.update_counts[player_id] == 0:
            # First update
            self.baselines[player_id] = reward
        else:
            # EMA update
            self.baselines[player_id] = (
                self.alpha * self.baselines[player_id] + (1 - self.alpha) * reward
            )
        self.update_counts[player_id] += 1

    def compute_advantage(self, player_id: int, reward: float) -> float:
        """Compute advantage for player by subtracting role-specific baseline."""
        return reward - self.baselines[player_id]

    def get_stats(self) -> Dict:
        """Get baseline statistics for logging."""
        return {
            "baseline_player_0": self.baselines[0],
            "baseline_player_1": self.baselines[1],
            "updates_player_0": self.update_counts[0],
            "updates_player_1": self.update_counts[1],
        }


def extract_move_from_response(response: str) -> int:
    """Extract move from model response following SPIRAL's action extraction."""
    # Look for answer tags first
    answer_match = re.search(r"<answer>\s*\[(\d+)\]\s*</answer>", response)
    if answer_match:
        try:
            return int(answer_match.group(1))
        except ValueError:
            pass

    # Fallback: look for numbers in brackets
    bracket_match = re.search(r"\[(\d+)\]", response)
    if bracket_match:
        try:
            return int(bracket_match.group(1))
        except ValueError:
            pass

    # Fallback: look for any digit 0-8
    digit_match = re.search(r"\b([0-8])\b", response)
    if digit_match:
        try:
            return int(digit_match.group(1))
        except ValueError:
            pass

    # Random fallback
    return random.randint(0, 8)


def play_self_play_game(model, tokenizer, device) -> GameTrajectory:
    """
    Play a single self-play game following SPIRAL's framework.
    Key features:
    - Shared policy plays both roles
    - Role conditioning through system prompts
    - Turn-based zero-sum dynamics
    """
    game = TicTacToeGame()
    player_trajectories = {0: [], 1: []}

    while not game.game_over:
        current_player = game.current_player

        # Get role-conditioned observation
        observation = game.get_observation(current_player)

        # Generate response using shared policy with role conditioning
        inputs = tokenizer(
            observation, return_tensors="pt", truncation=True, max_length=512
        )
        if device.type == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        # Extract move
        position = extract_move_from_response(response)
        valid_moves = game.get_valid_moves()

        # Check if move is valid
        is_valid = position in valid_moves
        if not is_valid and valid_moves:
            position = random.choice(valid_moves)  # Fallback to random valid move

        # Store trajectory data for this player
        player_trajectories[current_player].append(
            {
                "observation": observation,
                "response": response,
                "action": position,
                "is_valid": is_valid,
                "turn": game.turn_count,
            }
        )

        # Make move
        move_successful, game_over = game.make_move(current_player, position)

        if not move_successful:
            # Invalid move penalty
            game.game_over = True
            game.winner = 1 - current_player  # Opponent wins
            break

        # Safety check
        if game.turn_count > 20:
            game.game_over = True
            game.winner = "Draw"
            break

    # Get final rewards
    final_rewards = game.get_rewards()

    return GameTrajectory(
        player_trajectories=player_trajectories,
        final_rewards=final_rewards,
        game_length=game.turn_count,
    )


def compute_policy_gradient_loss(
    model,
    tokenizer,
    trajectories: List[GameTrajectory],
    rae: RoleConditionedAdvantageEstimation,
    device,
) -> torch.Tensor:
    """
    Compute policy gradient loss using SPIRAL's RAE method with memory optimization.
    """
    total_loss = torch.tensor(0.0, device=device)  # Changed to torch.Tensor
    num_updates = 0

    # Process trajectories in smaller chunks to avoid memory accumulation
    for trajectory in trajectories:
        for player_id in [0, 1]:
            if not trajectory.player_trajectories[player_id]:
                continue

            player_reward = trajectory.final_rewards[player_id]

            # Update RAE baseline for this player
            rae.update_baseline(player_id, player_reward)

            # Compute advantage using RAE
            advantage = rae.compute_advantage(player_id, player_reward)
            # Convert advantage to a detached PyTorch tensor
            advantage_tensor = torch.tensor(
                advantage, device=device, dtype=torch.float32
            ).detach()

            # Process moves one at a time to minimize memory usage
            trajectory_loss = torch.tensor(
                0.0, device=device
            )  # Changed to torch.Tensor
            for move_data in trajectory.player_trajectories[player_id]:
                observation = move_data["observation"]
                response = move_data["response"]

                # Tokenize with shorter max lengths to save memory
                inputs = tokenizer(
                    observation, return_tensors="pt", truncation=True, max_length=256
                )
                response_tokens = tokenizer(
                    response, return_tensors="pt", truncation=True, max_length=64
                )

                if device.type == "cuda":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    response_tokens = {
                        k: v.to(device) for k, v in response_tokens.items()
                    }

                # Skip if response is too short to avoid errors
                if response_tokens["input_ids"].shape[1] == 0:
                    continue

                # Forward pass
                with torch.no_grad():
                    # Get full sequence (prompt + response)
                    full_input_ids = torch.cat(
                        [inputs["input_ids"], response_tokens["input_ids"]], dim=1
                    )
                    full_attention_mask = torch.ones_like(full_input_ids)

                # Ensure correct dtype for input_ids (must be long for embedding layer)
                full_input_ids = full_input_ids.long()
                full_attention_mask = full_attention_mask.long()

                # Forward pass for loss computation (enable gradients for this specific pass)
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
                loss = (
                    -advantage_tensor * sequence_log_prob
                )  # Use the tensor version of advantage

                trajectory_loss += loss
                num_updates += 1

                # Clear intermediate tensors to free memory
                del outputs, logits, response_logits, log_probs, token_log_probs
                torch.cuda.empty_cache()

            total_loss += trajectory_loss

    return total_loss / max(num_updates, 1)


# Set up model cache directory
cache_dir = config["model"]["cache_dir"]
os.makedirs(cache_dir, exist_ok=True)

# Load model following existing pattern
if offline_mode:
    cached_models = glob.glob(os.path.join(cache_dir, "models--Qwen--Qwen2.5-*"))

    preferred_model = None
    fallback_model = None

    for cached_model_path in cached_models:
        model_name = (
            os.path.basename(cached_model_path)
            .replace("models--", "")
            .replace("--", "/")
        )
        if "1.5B" in model_name:
            preferred_model = model_name
            break
        elif "3B" in model_name:
            fallback_model = model_name

    if preferred_model:
        model_id = preferred_model
        print(f"🎯 Using preferred cached model: {model_id} (better memory efficiency)")
    elif fallback_model:
        model_id = fallback_model
        print(
            f"🔄 Using fallback cached model: {model_id} (will use aggressive memory settings)"
        )
    else:
        model_id = config["model"]["id"]
        print(f"⚠️  No cached models found, attempting: {model_id}")
else:
    model_id = config["model"]["id"]

print(f"📥 Loading model for SPIRAL self-play: {model_id}")

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

# Initialize SPIRAL components
rae = RoleConditionedAdvantageEstimation(alpha=config["training"]["rae_alpha"])
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

# Training parameters following SPIRAL paper concepts
num_steps = config["training"]["num_steps"]

# Adaptive batch size based on GPU memory
if platform_info["gpu_type"] == "V100":
    games_per_step = config["training"]["games_per_step_v100"]
    print("🔧 Using V100 memory-optimized settings: {games_per_step} games per step")
else:
    games_per_step = config["training"]["games_per_step_other"]
    print("🔧 Using conservative memory settings: {games_per_step} games per step")

print(
    f"🎮 Starting SPIRAL self-play training: {num_steps} steps, {games_per_step} games per step"
)

# Initialize wandb run (only if W&B is enabled by user)
if WANDB_ENABLED:
    # Create human-readable timestamp: Jul31_2025_14h30m
    timestamp = datetime.datetime.now().strftime("%b%d_%Y_%Hh%Mm")
    project_name = f"{config['wandb']['project_name_prefix']}-{timestamp}"
    wandb.init(project=project_name)
    print(f"✅ Initialized W&B run: {wandb.run.name} (Project: {project_name}, Offline mode: {offline_mode})")

# Run initial MBPP evaluation if enabled
if config["evaluation"].get("enabled_initial", True) and mbpp_evaluator.config.enabled:
    print("🧪 Running initial MBPP evaluation...")
    initial_results = mbpp_evaluator.evaluate_model(
        model, tokenizer, step=0, phase="initial"
    )

    if WANDB_ENABLED and wandb.run and initial_results.get("enabled", False):
        wandb.log(
            {
                "mbpp_eval/initial_pass_rate": initial_results["pass_rate"],
                "mbpp_eval/initial_problems_passed": initial_results["problems_passed"],
                "mbpp_eval/initial_total_problems": initial_results["total_problems"],
                "mbpp_eval/initial_eval_time": initial_results["eval_time_seconds"],
            }
        )

# Training loop
for step in range(num_steps):
    print(f"\n🎯 Step {step + 1}/{num_steps}")

    # Collect trajectories through self-play
    trajectories = []
    games_won = {0: 0, 1: 0, "Draw": 0}
    total_invalid_moves = 0

    for game_idx in range(games_per_step):
        trajectory = play_self_play_game(model, tokenizer, device)
        trajectories.append(trajectory)

        # Track statistics
        winner = None
        for player_id, reward in trajectory.final_rewards.items():
            if reward > 0:
                winner = player_id
                break

        if winner is not None:
            games_won[winner] += 1
        else:
            games_won["Draw"] += 1

        # Count invalid moves
        for player_id in [0, 1]:
            for move in trajectory.player_trajectories[player_id]:
                if not move["is_valid"]:
                    total_invalid_moves += 1

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
    avg_game_length = np.mean([t.game_length for t in trajectories])

    stats = {
        "step": step,
        "loss": loss.item(),
        "games_won_player_0": games_won[0],
        "games_won_player_1": games_won[1],
        "games_drawn": games_won["Draw"],
        "invalid_moves": total_invalid_moves,
        "avg_game_length": avg_game_length,
        **rae_stats,
    }

    print(f"📊 Loss: {loss.item():.4f}")
    print(
        f"🏆 Games - P0: {games_won[0]}, P1: {games_won[1]}, Draw: {games_won['Draw']}"
    )
    print(f"⚠️ Invalid moves: {total_invalid_moves}")
    print(f"📏 Avg game length: {avg_game_length:.1f}")
    print(
        f"📈 Baselines - P0: {rae_stats['baseline_player_0']:.3f}, P1: {rae_stats['baseline_player_1']:.3f}"
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

        if WANDB_ENABLED and wandb.run and interval_results.get("enabled", False):
            wandb.log(
                {
                    f"mbpp_eval/step_{step + 1}_pass_rate": interval_results[
                        "pass_rate"
                    ],
                    f"mbpp_eval/step_{step + 1}_problems_passed": interval_results[
                        "problems_passed"
                    ],
                    f"mbpp_eval/step_{step + 1}_total_problems": interval_results[
                        "total_problems"
                    ],
                    f"mbpp_eval/step_{step + 1}_eval_time": interval_results[
                        "eval_time_seconds"
                    ],
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

print("🏁 SPIRAL self-play training completed!")

# Run final MBPP evaluation if enabled
if config["evaluation"].get("enabled_final", True) and mbpp_evaluator.config.enabled:
    print("🧪 Running final MBPP evaluation...")
    final_results = mbpp_evaluator.evaluate_model(
        model, tokenizer, step=num_steps, phase="final"
    )

    if WANDB_ENABLED and wandb.run and final_results.get("enabled", False):
        wandb.log(
            {
                "mbpp_eval/final_pass_rate": final_results["pass_rate"],
                "mbpp_eval/final_problems_passed": final_results["problems_passed"],
                "mbpp_eval/final_total_problems": final_results["total_problems"],
                "mbpp_eval/final_eval_time": final_results["eval_time_seconds"],
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
