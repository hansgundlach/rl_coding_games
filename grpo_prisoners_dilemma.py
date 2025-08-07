#!/usr/bin/env python3
"""
GRPO Iterated Prisoner's Dilemma Training

GRPO-based self-play training where an LLM competes by submitting code strategies 
for the iterated prisoner's dilemma. The model plays against a frozen copy of itself
that gets refreshed every N updates.

Key Features:
- Single LLM (e.g., Qwen3-1.7B + LoRA) with self-play
- LLMs submit code strategies, not direct actions  
- Reward based on tournament wins/losses
- -1 reward for code execution failures
- WSLS bot opponents interspersed for robustness
- Action noise for more realistic gameplay
- Frozen opponent refreshed every N steps
- CPU parallelism for game simulation
"""

# Set environment variable to avoid tokenizers warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import re
import wandb
import sys
import datetime
import glob
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import json
import yaml
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor

print("🚀 Starting GRPO Prisoner's Dilemma Training...")
print("📋 Initializing components...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="GRPO Prisoner's Dilemma Training")
parser.add_argument(
    "--config",
    type=str,
    default="configs/grpo_prisoners_dilemma.yaml",
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

        # Validate that the config path exists (except for the final key)
        for i, k in enumerate(keys[:-1]):
            if k not in current:
                print(f"❌ ERROR: Invalid config override path '{key}'")
                print(f"   Key '{k}' does not exist in config at level {i+1}")
                print(f"   Available keys at this level: {list(current.keys())}")
                raise ValueError(f"Invalid config override path: {key}")
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set global environment variables for transformers library
if offline_mode:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    print("🌐 Set offline mode environment variables")

# Add project path to sys.path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("📦 Loading utility modules...")
from utils.env_loader import get_api_key
from utils.seed_manager import SeedManager
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
        print("⚠️ No W&B API key found, continuing without logging")
else:
    print("🚫 Skipping W&B login (W&B is disabled by user).")

# Initialize comprehensive seed management
print("🎲 Setting up seed management...")
seed_manager = SeedManager.from_config(config)
seed_manager.seed_everything()

# Initialize MBPP evaluator with consolidated config
print("🧪 Setting up MBPP evaluator...")

# Import the configuration helper
from evaluation.configs.loader import get_platform_specific_config

# Create evaluation config from main config - RESPECT CONFIG FILE VALUES
eval_config_dict = config.get("evaluation", {}).copy()

# Remove keys not expected by EvalConfig constructor
eval_config_dict.pop("enabled_initial", None)
eval_config_dict.pop("enabled_final", None)
eval_config_dict.pop("enabled_interval", None)
eval_config_dict.pop("eval_interval_steps", None)

# Only add platform-specific adjustments for missing values (don't override config file)
platform_config = get_platform_specific_config()
for key, value in platform_config.items():
    if key not in eval_config_dict or eval_config_dict[key] is None:
        eval_config_dict[key] = value

# Create EvalConfig object from config file values
eval_config = EvalConfig(**eval_config_dict)

mbpp_evaluator = MBPPEvaluator(eval_config)

if not mbpp_evaluator.config.enabled:
    print("⚠️ MBPP evaluation disabled - dataset not found")
    print("💡 To enable evaluation, download MBPP dataset first:")
    print("   python -m evaluation.mbpp.evaluator")
else:
    print(
        f"✅ MBPP evaluation enabled with {mbpp_evaluator.config.num_questions} questions"
    )

# Initialize prisoner's dilemma game environment
print("🏗️ Setting up prisoner's dilemma game environment...")
game_env = IteratedPrisonersDilemma(config["game"])
print(f"🎮 Game: {game_env.get_game_name()}")
print(f"   Rounds per game: {config['game']['num_rounds']}")
print(f"   Noise enabled: {config['game']['noise_enabled']}")
if config["game"]["noise_enabled"]:
    print(f"   Noise probability: {config['game']['noise_prob']}")
print(f"   WSLS bot enabled: {config['game']['wsls_bot_enabled']}")
if config["game"]["wsls_bot_enabled"]:
    print(f"   WSLS bot probability: {config['game']['wsls_bot_prob']}")

@dataclass
class PrisonersReward:
    """Reward data for a prisoner's dilemma game."""
    game_id: int
    player1_reward: float
    player2_reward: float
    game_data: Dict[str, Any]

def generate_strategy_pair(
    main_model, opponent_model, tokenizer, device, game_env, step: int, game_idx: int
) -> Tuple[PlayerSubmission, PlayerSubmission]:
    """Generate strategy code for both players using main model vs opponent."""
    player_submissions = []
    
    # Player 1: Main model (being trained)
    # Player 2: Opponent model (frozen copy or WSLS bot)
    
    models = [main_model, opponent_model]
    
    for player_id in [0, 1]:
        # Get prompt for this player
        prompt = game_env.get_player_prompt(player_id, "player")
        
        # Generate response with the appropriate model
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        if device.type == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        model = models[player_id]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config["generation"]["max_new_tokens"],
                temperature=config["generation"]["temperature"],
                top_p=config["generation"]["top_p"],
                top_k=config["generation"]["top_k"],
                do_sample=config["generation"]["do_sample"],
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode the generated response
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        player_submissions.append(
            PlayerSubmission(player_id=player_id, code=generated_text)
        )

    return player_submissions[0], player_submissions[1]

def simulate_single_game(strategy_pair_with_env) -> 'GameTrajectory':
    """Simulate a single game with given strategies (CPU-bound, can be parallelized)."""
    player1_submission, player2_submission, game_env_copy, game_id = strategy_pair_with_env
    
    # Play the game
    game_result = game_env_copy.play_game([player1_submission, player2_submission])
    
    return GameTrajectory(
        game_id=game_id,
        player1_submission=player1_submission,
        player2_submission=player2_submission,
        game_result=game_result,
    )

@dataclass 
class GameTrajectory:
    """Complete trajectory for a single game."""
    game_id: int
    player1_submission: PlayerSubmission
    player2_submission: PlayerSubmission
    game_result: Any  # GameResult from game environment

# Load and setup models
print("🤖 Loading and setting up models...")

# Model configuration
model_id = config["model"]["id"]
cache_dir = config["model"]["cache_dir"]

print(f"📥 Loading model: {model_id}")
print(f"💾 Cache directory: {cache_dir}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id, cache_dir=cache_dir, local_files_only=offline_mode
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load main model
main_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_dir,
    local_files_only=offline_mode,
)

print(f"✅ Loaded main model with {sum(p.numel() for p in main_model.parameters()):,} parameters")

# Apply LoRA to main model
lora_config = LoraConfig(**config["lora"])
main_model = get_peft_model(main_model, lora_config)
main_model.print_trainable_parameters()

# Create initial frozen opponent (copy of main model)
print("🥶 Creating initial frozen opponent...")
opponent_model = copy.deepcopy(main_model)
for param in opponent_model.parameters():
    param.requires_grad = False
opponent_model.eval()

print("✅ Created frozen opponent model")

# Initialize GRPO trainer 
print("🏗️ Setting up GRPO trainer...")

# GRPO reward function
def prisoners_dilemma_reward_function(completions, **kwargs):
    """
    GRPO reward function that plays prisoner's dilemma games between main model and opponent.
    
    Args:
        completions: List of generated completions from main model (player 1)
        **kwargs: Additional arguments from GRPO trainer
        
    Returns:
        List of reward values for each completion
    """
    print(f"🎮 Playing {len(completions)} prisoner's dilemma games for reward calculation...")
    
    rewards = []
    game_stats = {
        "player1_wins": 0,
        "player2_wins": 0, 
        "ties": 0,
        "execution_failures": 0,
        "successful_games": 0,
        "wsls_bot_games": 0,
        "games_with_noise": 0,
    }
    
    # Generate opponent strategies for each game
    opponent_completions = []
    for i in range(len(completions)):
        # Generate opponent strategy with frozen model
        prompt = game_env.get_player_prompt(1, "player")  # Player 2 prompt
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if device.type == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = opponent_model.generate(
                **inputs,
                max_new_tokens=config["generation"]["max_new_tokens"],
                temperature=config["generation"]["temperature"],
                top_p=config["generation"]["top_p"],
                top_k=config["generation"]["top_k"],
                do_sample=config["generation"]["do_sample"],
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        opponent_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        opponent_completions.append(opponent_text)
    
    # Play games in parallel
    strategy_pairs = []
    for i, (main_completion, opp_completion) in enumerate(zip(completions, opponent_completions)):
        player1_sub = PlayerSubmission(player_id=0, code=main_completion)
        player2_sub = PlayerSubmission(player_id=1, code=opp_completion)
        # Create a fresh copy of game_env for each game to avoid threading issues
        game_env_copy = copy.deepcopy(game_env)
        strategy_pairs.append((player1_sub, player2_sub, game_env_copy, i))
    
    # Simulate games (potentially in parallel)
    parallel_games = config["training"].get("parallel_games", True)
    num_workers = config["training"].get("num_workers", None)
    
    if parallel_games and len(strategy_pairs) > 1:
        max_workers = num_workers or (os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            trajectories = list(executor.map(simulate_single_game, strategy_pairs))
    else:
        trajectories = [simulate_single_game(pair) for pair in strategy_pairs]
    
    # Calculate rewards
    for trajectory in trajectories:
        if trajectory.game_result.success:
            game_stats["successful_games"] += 1
            
            # Extract player rewards
            p1_reward = trajectory.game_result.player_rewards[0]
            p2_reward = trajectory.game_result.player_rewards[1]
            
            # Determine winner and assign GRPO reward
            if p1_reward > p2_reward:
                reward = 1.0  # Main model wins
                game_stats["player1_wins"] += 1
            elif p2_reward > p1_reward:
                reward = -1.0  # Opponent wins
                game_stats["player2_wins"] += 1
            else:
                reward = 0.0  # Tie
                game_stats["ties"] += 1
            
            # Track special game features
            game_data = trajectory.game_result.game_data
            if game_data.get("wsls_bot_used", False):
                game_stats["wsls_bot_games"] += 1
            if game_data.get("noise_rounds", 0) > 0:
                game_stats["games_with_noise"] += 1
                
        else:
            # Code execution failure - negative reward
            reward = -1.0
            game_stats["execution_failures"] += 1
            
        rewards.append(reward)
    
    # Store stats for logging
    prisoners_dilemma_reward_function.last_game_stats = game_stats
    
    # Show some debug info
    debug_config = config.get("debug", {})
    show_detailed_games = debug_config.get("show_detailed_games", 0)
    
    if show_detailed_games > 0:
        print(f"🎯 Showing details for first {min(show_detailed_games, len(trajectories))} games:")
        for i in range(min(show_detailed_games, len(trajectories))):
            trajectory = trajectories[i]
            reward = rewards[i]
            print(f"\n🎮 Game {i+1}: Reward = {reward:+.1f}")
            
            if trajectory.game_result.success:
                p1_reward = trajectory.game_result.player_rewards[0]
                p2_reward = trajectory.game_result.player_rewards[1]
                print(f"   Player scores: P1={p1_reward}, P2={p2_reward}")
                
                game_data = trajectory.game_result.game_data
                features = []
                if game_data.get("wsls_bot_used", False):
                    features.append("🤖 WSLS Bot")
                if game_data.get("noise_rounds", 0) > 0:
                    features.append(f"🎲 Noise ({game_data['noise_rounds']} rounds)")
                if features:
                    print(f"   Features: {', '.join(features)}")
            else:
                print(f"   ❌ Game failed: {trajectory.game_result.error}")
    
    print(f"🏆 Game results: {game_stats['player1_wins']} wins, {game_stats['player2_wins']} losses, {game_stats['ties']} ties")
    print(f"💥 Execution failures: {game_stats['execution_failures']}/{len(completions)}")
    
    return rewards

# Create GRPO config
grpo_config = GRPOConfig(**config["grpo_config"])

# Create dataset for GRPO (prompts for player 1)
print("📚 Creating training dataset...")
prompts = []
for i in range(config["dataset"]["size"]):
    prompt = game_env.get_player_prompt(0, "player")  # Always player 1 prompt
    prompts.append(prompt)

dataset = Dataset.from_dict({"prompt": prompts})
print(f"✅ Created dataset with {len(dataset)} prompts")

# Create GRPO trainer
print("🎯 Initializing GRPO trainer...")
grpo_trainer = GRPOTrainer(
    model=main_model,
    reward_funcs=[prisoners_dilemma_reward_function],
    args=grpo_config,
    train_dataset=dataset,
)
print("✅ GRPO trainer initialized")

# Training parameters
num_steps = config["training"]["num_steps"]
opponent_refresh_steps = config["training"]["opponent_refresh_steps"]

# Adaptive batch size based on GPU memory
if platform_info["gpu_type"] == "V100":
    games_per_step = config["training"]["games_per_step_v100"]
    print(f"🔧 Using V100 memory-optimized settings: {games_per_step} games per step")
else:
    games_per_step = config["training"]["games_per_step_other"]
    print(f"🔧 Using standard memory settings: {games_per_step} games per step")

print(f"🎮 Starting GRPO prisoner's dilemma training: {num_steps} steps")
print(f"🥶 Opponent refresh every {opponent_refresh_steps} steps")

# Initialize wandb run (only if W&B is enabled by user)
if WANDB_ENABLED:
    if not offline_mode:
        # Online mode: Create timestamped project name like grpo_code_game
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        project_name = f"{config['wandb']['project_name_prefix']}-{timestamp}"
        wandb.init(
            project=project_name, config={**config, **seed_manager.get_seed_info()}
        )
        print(f"✅ Initialized W&B run: {wandb.run.name} (Project: {project_name})")
    else:
        # Offline mode: Use consistent project name without timestamp
        project_name = config["wandb"]["project_name_prefix"]
        wandb.init(
            project=project_name,
            config={**config, **seed_manager.get_seed_info()},
            mode="offline",
        )
        print(
            f"✅ Initialized W&B run (offline): {wandb.run.name} (Project: {project_name})"
        )

# Run initial MBPP evaluation if enabled
if config["evaluation"].get("enabled_initial", True) and mbpp_evaluator.config.enabled:
    print("🧪 Running initial MBPP evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("initial")
    initial_results = mbpp_evaluator.evaluate_model(
        main_model, tokenizer, step=0, phase="initial"
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

print("\n🏋️ Starting GRPO training loop...")

# Training loop
for step in range(num_steps):
    print(f"\n🎯 Step {step + 1}/{num_steps}")
    
    # Refresh opponent model every N steps
    if step > 0 and step % opponent_refresh_steps == 0:
        print(f"🔄 Refreshing opponent model (step {step + 1})...")
        opponent_model = copy.deepcopy(main_model)
        for param in opponent_model.parameters():
            param.requires_grad = False
        opponent_model.eval()
        print("✅ Opponent model refreshed with latest weights")
    
    # Run one step of GRPO training
    print("🏋️ Running GRPO training step...")
    grpo_trainer.train()
    
    # Log training metrics
    if WANDB_ENABLED and wandb.run:
        # Get last game stats from reward function
        if hasattr(prisoners_dilemma_reward_function, 'last_game_stats'):
            stats = prisoners_dilemma_reward_function.last_game_stats
            
            wandb.log({
                "training/step": step + 1,
                "training/player1_wins": stats["player1_wins"],
                "training/player2_wins": stats["player2_wins"],
                "training/ties": stats["ties"],
                "training/win_rate": stats["player1_wins"] / max(1, stats["successful_games"]),
                "training/execution_failures": stats["execution_failures"],
                "training/success_rate": stats["successful_games"] / max(1, len(dataset)),
                "training/wsls_bot_games": stats["wsls_bot_games"],
                "training/games_with_noise": stats["games_with_noise"],
            })
    
    # Run interval MBPP evaluation if enabled
    if (
        config["evaluation"].get("enabled_interval", False) 
        and mbpp_evaluator.config.enabled
        and config["evaluation"].get("eval_interval_steps", 10) > 0
        and (step + 1) % config["evaluation"]["eval_interval_steps"] == 0
    ):
        print(f"🧪 Running interval MBPP evaluation at step {step + 1}...")
        # Seed for consistent evaluation
        seed_manager.seed_for_evaluation_auto(f"interval_step_{step + 1}")
        interval_results = mbpp_evaluator.evaluate_model(
            main_model, tokenizer, step=step + 1, phase="interval"
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
    
    # Save checkpoint
    if (step + 1) % config["training"]["save_interval"] == 0:
        checkpoint_dir = f"{config['training']['checkpoint_dir']}/step_{step + 1}"
        print(f"💾 Saving checkpoint at step {step + 1}...")
        main_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"✅ Checkpoint saved to {checkpoint_dir}")

print("\n🏁 GRPO Prisoner's Dilemma training completed!")

# Run final MBPP evaluation if enabled
if config["evaluation"].get("enabled_final", True) and mbpp_evaluator.config.enabled:
    print("🧪 Running final MBPP evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("final")
    final_results = mbpp_evaluator.evaluate_model(
        main_model, tokenizer, step=num_steps, phase="final"
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

# Save final model
final_checkpoint_dir = f"{config['training']['checkpoint_dir']}/final"
print(f"💾 Saving final model...")
main_model.save_pretrained(final_checkpoint_dir)
tokenizer.save_pretrained(final_checkpoint_dir)
print(f"✅ Final model saved to {final_checkpoint_dir}")

if WANDB_ENABLED and wandb.run:
    wandb.finish()

print("🎉 Training completed successfully!")