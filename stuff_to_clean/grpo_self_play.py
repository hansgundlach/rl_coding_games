#!/usr/bin/env python3
"""
GRPO Self-Play Training - Model plays Tic-Tac-Toe against itself
Simplified implementation based on SPIRAL paper approach.
"""

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import GRPOConfig, GRPOTrainer
import re
import io
import contextlib
import wandb
import sys
import os
import datetime
import glob
import random
import numpy as np
from typing import List, Tuple, Optional


def detect_platform_and_gpu():
    """Auto-detect platform and GPU capabilities for environment-specific settings."""
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

# Set global environment variables for transformers library
if offline_mode:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    print("✅ Set global offline mode for transformers and wandb")

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.env_loader import get_api_key

# Initialize wandb with API key from environment (skip if offline)
if not offline_mode:
    wandb_key = get_api_key("wandb", required=False)
    if wandb_key:
        wandb.login(key=wandb_key)
        print("✓ Logged into W&B using environment variable")
    else:
        print("⚠️ No W&B API key found, continuing without logging")
else:
    print("🚫 Skipping W&B login (offline mode)")


class TicTacToeGame:
    """Simple Tic-Tac-Toe game implementation."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to initial state."""
        self.board = [" "] * 9  # 9 positions: 0-8
        self.current_player = "X"  # X always starts
        self.game_over = False
        self.winner = None
        self.moves_played = []

    def get_board_string(self) -> str:
        """Get a string representation of the board."""
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
        return board_str

    def get_prompt(self) -> str:
        """Generate prompt for current game state."""
        prompt = f"""You are playing Tic-Tac-Toe. You are player {self.current_player}.

{self.get_board_string()}

Available positions: {self.get_valid_moves()}

Choose your move by responding with just the position number (0-8).
Your move: """
        return prompt

    def get_valid_moves(self) -> List[int]:
        """Get list of valid moves (empty positions)."""
        return [i for i in range(9) if self.board[i] == " "]

    def make_move(self, position: int) -> bool:
        """Make a move at the given position. Returns True if valid move."""
        if (
            position < 0
            or position > 8
            or self.board[position] != " "
            or self.game_over
        ):
            return False

        self.board[position] = self.current_player
        self.moves_played.append((self.current_player, position))

        # Check for winner
        if self.check_winner():
            self.winner = self.current_player
            self.game_over = True
        elif len(self.get_valid_moves()) == 0:
            # Draw
            self.game_over = True
            self.winner = "Draw"
        else:
            # Switch players
            self.current_player = "O" if self.current_player == "X" else "X"

        return True

    def check_winner(self) -> bool:
        """Check if current player has won."""
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
            if all(self.board[i] == self.current_player for i in combo):
                return True
        return False

    def get_rewards(self) -> dict:
        """Get rewards for both players based on game outcome."""
        if not self.game_over:
            return {"X": 0, "O": 0}

        if self.winner == "X":
            return {"X": 1, "O": -1}
        elif self.winner == "O":
            return {"X": -1, "O": 1}
        else:  # Draw
            return {"X": 0, "O": 0}


class GameTrajectory:
    """Store trajectory data for a single game."""

    def __init__(self):
        self.moves = []  # List of (player, prompt, response, position, valid)

    def add_move(
        self, player: str, prompt: str, response: str, position: int, is_valid: bool
    ):
        """Add a move to the trajectory."""
        self.moves.append(
            {
                "player": player,
                "prompt": prompt,
                "response": response,
                "position": position,
                "is_valid": is_valid,
            }
        )


def play_self_play_game(
    model, tokenizer, max_length: int = 256
) -> Tuple[GameTrajectory, dict]:
    """
    Play a single self-play game where the model plays against itself.

    Returns:
        GameTrajectory: Trajectory of the game
        dict: Final rewards for both players
    """
    game = TicTacToeGame()
    trajectory = GameTrajectory()

    while not game.game_over:
        # Get current game state and prompt
        prompt = game.get_prompt()
        current_player = game.current_player

        # Generate model response
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        # Extract move from response
        position = extract_move(response)

        # Check if move is valid
        valid_moves = game.get_valid_moves()
        is_valid = position in valid_moves

        # Make the move
        if is_valid:
            success = game.make_move(position)
        else:
            # Invalid move - pick random valid move
            position = random.choice(valid_moves) if valid_moves else 0
            success = game.make_move(position)
            is_valid = False

        # Store in trajectory
        trajectory.add_move(current_player, prompt, response, position, is_valid)

        # Safety check to prevent infinite loops
        if len(trajectory.moves) > 20:
            break

    rewards = game.get_rewards()
    return trajectory, rewards


def extract_move(response: str) -> int:
    """Extract move position from model response."""
    # Look for numbers 0-8 in the response
    import re

    numbers = re.findall(r"\b[0-8]\b", response)

    if numbers:
        return int(numbers[0])

    # Fallback: return random valid position
    return random.randint(0, 8)


# Set up model cache directory
cache_dir = "./model_cache"
os.makedirs(cache_dir, exist_ok=True)

# Use exactly the model specified in config
model_id = config["model"]["id"]
print(f"📥 Using model from config: {model_id}")

print(f"Loading trainable model: {model_id}")

model1 = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_dir,
    local_files_only=offline_mode,
)
tokenizer1 = AutoTokenizer.from_pretrained(
    model_id, cache_dir=cache_dir, local_files_only=offline_mode
)
# Ensure tokenizer has pad token
if tokenizer1.pad_token is None:
    tokenizer1.pad_token = tokenizer1.eos_token

# Load LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)
model1 = get_peft_model(model1, lora_config)
print(model1.print_trainable_parameters())


def self_play_reward_function(completions, **kwargs):
    """
    Reward function for self-play training.

    The model plays complete games against itself and gets rewarded based on outcomes.
    This is different from single-turn rewards - we need to play full games.
    """
    print("🎮 Starting self-play games...")

    # We'll generate a smaller number of games since each game has multiple moves
    num_games = min(len(completions) // 4, 8)  # Each game generates ~4-9 moves
    all_rewards = []

    games_won = 0
    games_lost = 0
    games_drawn = 0
    invalid_moves = 0

    for game_idx in range(num_games):
        print(f"🎯 Playing game {game_idx + 1}/{num_games}")

        # Play a self-play game
        trajectory, final_rewards = play_self_play_game(model1, tokenizer1)

        # Process each move in the trajectory
        for move_idx, move in enumerate(trajectory.moves):
            player = move["player"]
            is_valid = move["is_valid"]

            # Calculate reward for this move
            if not is_valid:
                # Penalty for invalid moves
                reward = -0.5
                invalid_moves += 1
            else:
                # Reward based on final game outcome, discounted by move number
                final_reward = final_rewards[player]
                # Earlier moves get less reward (they contributed less to final outcome)
                discount = 0.9 ** (len(trajectory.moves) - move_idx - 1)
                reward = final_reward * discount

            all_rewards.append(reward)

            # Debug output for first few moves
            if len(all_rewards) <= 3:
                print("=" * 50)
                print(f"Game {game_idx + 1}, Move {move_idx + 1}:")
                print(f"Player: {player}")
                print(f"Response: {move['response'][:100]}...")
                print(f"Position: {move['position']}")
                print(f"Valid: {is_valid}")
                print(f"Final game outcome: {final_rewards}")
                print(f"Move reward: {reward:.2f}")
                print("=" * 50)

        # Track game outcomes
        if final_rewards["X"] > final_rewards["O"]:
            games_won += 1 if trajectory.moves[0]["player"] == "X" else 0
            games_lost += 1 if trajectory.moves[0]["player"] == "O" else 0
        elif final_rewards["X"] < final_rewards["O"]:
            games_won += 1 if trajectory.moves[0]["player"] == "O" else 0
            games_lost += 1 if trajectory.moves[0]["player"] == "X" else 0
        else:
            games_drawn += 1

    # Pad rewards to match completion count
    while len(all_rewards) < len(completions):
        all_rewards.append(0.0)

    # Truncate if we have too many rewards
    all_rewards = all_rewards[: len(completions)]

    # Log metrics to wandb
    if wandb.run:
        avg_reward = np.mean(all_rewards) if all_rewards else 0

        wandb.log(
            {
                "selfplay/avg_reward": avg_reward,
                "selfplay/games_played": num_games,
                "selfplay/games_won": games_won,
                "selfplay/games_lost": games_lost,
                "selfplay/games_drawn": games_drawn,
                "selfplay/invalid_moves": invalid_moves,
                "selfplay/total_moves": len(all_rewards),
            }
        )

    print(f"🏁 Completed {num_games} games, generated {len(all_rewards)} rewards")
    print(f"📊 Games: {games_won} won, {games_lost} lost, {games_drawn} drawn")
    print(f"⚠️ Invalid moves: {invalid_moves}")

    return all_rewards


# Create a dummy dataset (we don't use traditional prompts in self-play)
dummy_prompt = "Play tic-tac-toe"
dataset = Dataset.from_dict(
    {
        "prompt": [dummy_prompt]
        * 500  # Smaller dataset since each generates multiple moves
    }
)

print(f"Created dataset: {dataset}")

# Training arguments - adjust for GPU memory capacity and model size
print("Setting up GRPO training for self-play...")

# Adaptive settings based on GPU type and model size
if platform_info["gpu_type"] == "V100":
    if "3B" in model_id:
        # V100 with 3B model - very aggressive memory reduction
        batch_size = 1
        gradient_accumulation_steps = 8
        max_prompt_length = 256
        max_completion_length = 128
        print("🔧 Using V100 + 3B model settings (very aggressive memory reduction)")
    else:
        # V100 with 1.5B model - moderate memory reduction
        batch_size = 2
        gradient_accumulation_steps = 4
        max_prompt_length = 512
        max_completion_length = 256
        print("🔧 Using V100 + 1.5B model settings (moderate memory reduction)")
else:
    # A100 or other GPUs - standard settings
    batch_size = 4
    gradient_accumulation_steps = 2
    max_prompt_length = 512
    max_completion_length = 256
    print("🔧 Using standard memory settings for A100/other GPUs")

training_args = GRPOConfig(
    output_dir="checkpoints/grpo_self_play",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    num_generations=2,  # GRPO requires minimum 2
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=platform_info["supports_bf16"],  # Auto-configured based on GPU type
    fp16=not platform_info["supports_bf16"],  # Use fp16 if bf16 not supported
    gradient_checkpointing=(
        True if platform_info["gpu_type"] == "V100" else False
    ),  # Memory optimization for V100
    dataloader_pin_memory=(
        False if platform_info["gpu_type"] == "V100" else True
    ),  # Memory optimization for V100
    report_to=["wandb"] if not offline_mode else [],
    remove_unused_columns=False,
    logging_steps=1,
)

# Fix model config path for GRPOTrainer if in offline mode
if offline_mode:
    # Point the model config to actual cached snapshot directory so GRPOTrainer can find tokenizer files locally
    cached_model_dirs = glob.glob(
        os.path.join(
            cache_dir, f"models--{model_id.replace('/', '--')}", "snapshots", "*"
        )
    )
    if cached_model_dirs:
        model1.config._name_or_path = cached_model_dirs[0]
        print(f"🔧 Set model config path to: {cached_model_dirs[0]}")

# Create trainer
trainer = GRPOTrainer(
    model=model1,
    reward_funcs=[self_play_reward_function],
    args=training_args,
    train_dataset=dataset,
)

print("Starting GRPO self-play training...")

# Initialize wandb run (only if not in offline mode)
if not offline_mode:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb.init(project=f"qwen-self-play-grpo-{timestamp}")
else:
    print("🚫 Skipping wandb initialization (offline mode)")

# Train model
trainer.train()

print("Self-play training completed!")
if not offline_mode:
    wandb.finish()
else:
    print("✅ Training completed (offline mode - no wandb to finish)")
