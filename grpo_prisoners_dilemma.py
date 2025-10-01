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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer  # type: ignore
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

        # Validate that the config path existsx (except for the final key)
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set global environment variables for transformers library and W&B
if WANDB_ENABLED:  # Check if W&B is enabled by user
    if offline_mode:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["WANDB_MODE"] = "offline"

        # Set WANDB_RUN_ID before importing wandb to control offline directory name
        timestamp = datetime.datetime.now().strftime("%b%d_%Y_%Hh%Mm")
        run_id = f"grpo-prisoners-dilemma-{timestamp.replace('_', '-').replace('h', 'h-').replace('m', 'm')}"
        os.environ["WANDB_RUN_ID"] = run_id
        print(f"🔧 Set WANDB_RUN_ID for offline mode: {run_id}")

        print("✅ Set global offline mode for transformers and wandb")
    else:
        # For online mode, ensure offline flags are not set
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        os.environ.pop("WANDB_MODE", None)
else:  # If WANDB_ENABLED is False, explicitly disable W&B
    os.environ["WANDB_MODE"] = "disabled"
    print("🚫 W&B logging explicitly disabled by user configuration.")

# Add project path to sys.path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("📦 Loading utility modules...")
from utils.env_loader import get_api_key
from utils.seed_manager import SeedManager
from evaluation.mbpp.evaluator import MBPPEvaluator, EvalConfig
from evaluation.alignment.evaluator import AlignmentEvaluator, AlignmentEvalConfig
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

if not config["evaluation"].get("enabled", True):
    print("⚠️ MBPP coding evaluation disabled by configuration")
elif not mbpp_evaluator.config.enabled:
    print("⚠️ MBPP evaluation disabled - dataset not found")
    print("💡 To enable evaluation, download MBPP dataset first:")
    print("   python -m evaluation.mbpp.evaluator")
else:
    print(
        f"✅ MBPP evaluation enabled with {mbpp_evaluator.config.num_questions} questions"
    )

# Initialize alignment evaluator
print("🧪 Setting up alignment evaluator...")
alignment_eval_config_dict = config.get("alignment_evaluation", {})
alignment_eval_config = AlignmentEvalConfig(
    enabled=alignment_eval_config_dict.get("enabled", True),
    eval_at_start=alignment_eval_config_dict.get("enabled_initial", True),
    eval_at_end=alignment_eval_config_dict.get("enabled_final", True),
    eval_interval_steps=alignment_eval_config_dict.get("eval_interval_steps", None),
    temperature=alignment_eval_config_dict.get("temperature", 0.7),
    do_sample=alignment_eval_config_dict.get("do_sample", True),
    max_new_tokens=alignment_eval_config_dict.get("max_new_tokens", 512),
    results_dir=alignment_eval_config_dict.get("results_dir", "./eval_results/alignment"),
    verbose=alignment_eval_config_dict.get("verbose", True)
)

alignment_evaluator = AlignmentEvaluator(alignment_eval_config)

if alignment_evaluator.config.enabled:
    print("✅ Alignment evaluation enabled")
else:
    print("⚠️ Alignment evaluation disabled")

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

        generation_kwargs = {
            "max_new_tokens": config["generation"]["max_new_tokens"],
            "temperature": config["generation"]["temperature"],
            "top_p": config["generation"]["top_p"],
            "do_sample": config["generation"]["do_sample"],
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        # Only add top_k if it's specified and positive
        if "top_k" in config["generation"] and config["generation"]["top_k"] > 0:
            generation_kwargs["top_k"] = config["generation"]["top_k"]

        model = models[player_id]
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        # Decode the generated response
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        # Extract strategy code and validate before submitting
        extracted_code = game_env.extract_code_from_response(generated_text)
        is_valid, validation_error = game_env.validate_submission(
            extracted_code, player_id, "player"
        )

        player_submissions.append(
            PlayerSubmission(
                player_id=player_id,
                role="player",
                prompt=prompt,
                response=generated_text,
                extracted_code=extracted_code,
                compilation_success=is_valid,
                compilation_error=(None if is_valid else validation_error),
            )
        )

    return player_submissions[0], player_submissions[1]


def simulate_single_game(strategy_pair_with_env) -> "GameTrajectory":
    """Simulate a single game with given strategies (CPU-bound, can be parallelized)."""
    player1_submission, player2_submission, game_env_copy, game_id = (
        strategy_pair_with_env
    )

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


# Set up model cache directory
cache_dir = config["model"]["cache_dir"]
os.makedirs(cache_dir, exist_ok=True)

# Use exactly the model specified in config
model_id = config["model"]["id"]
print(f"📥 Using model from config: {model_id}")

# Set up quantization if enabled
quantization_config = None
if config["model"].get("quantization", {}).get("enabled", False):
    print("🔧 Setting up quantization with bitsandbytes...")
    quantization_settings = config["model"]["quantization"]

    # Validate quantization settings
    if quantization_settings.get("load_in_4bit", False) and quantization_settings.get(
        "load_in_8bit", False
    ):
        raise ValueError(
            "Cannot enable both 4-bit and 8-bit quantization simultaneously"
        )

    if not quantization_settings.get(
        "load_in_4bit", False
    ) and not quantization_settings.get("load_in_8bit", False):
        raise ValueError(
            "Quantization enabled but neither 4-bit nor 8-bit quantization specified"
        )

    # Create BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quantization_settings.get("load_in_4bit", False),
        load_in_8bit=quantization_settings.get("load_in_8bit", False),
        bnb_4bit_compute_dtype=getattr(
            torch, quantization_settings.get("bnb_4bit_compute_dtype", "float16")
        ),
        bnb_4bit_use_double_quant=quantization_settings.get(
            "bnb_4bit_use_double_quant", True
        ),
        bnb_4bit_quant_type=quantization_settings.get("bnb_4bit_quant_type", "nf4"),
    )

    print(
        f"✅ Quantization config: 4-bit={quantization_settings.get('load_in_4bit', False)}, "
        f"8-bit={quantization_settings.get('load_in_8bit', False)}"
    )
else:
    print("📦 Loading model without quantization")

# Load model with optional quantization
model_kwargs = {
    "torch_dtype": "auto",
    "device_map": "auto",
    "cache_dir": cache_dir,
    "local_files_only": offline_mode,
}

if quantization_config is not None:
    model_kwargs["quantization_config"] = quantization_config

main_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(
    model_id, cache_dir=cache_dir, local_files_only=offline_mode
)

# Ensure tokenizer has pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(
    f"✅ Loaded main model with {sum(p.numel() for p in main_model.parameters()):,} parameters"
)

# Add LoRA for efficient training
print("🔧 Setting up LoRA configuration...")

# Check if we're using 4-bit quantization (QLoRA)
is_4bit_quantized = config["model"].get("quantization", {}).get(
    "enabled", False
) and config["model"]["quantization"].get("load_in_4bit", False)

if is_4bit_quantized:
    print("🔧 Using QLoRA configuration for 4-bit quantization...")
    # QLoRA requires specific settings
    lora_config = LoraConfig(
        task_type=config["lora"]["task_type"],
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        # QLoRA specific settings
        inference_mode=False,
        bias="none",
    )
else:
    print("🔧 Using standard LoRA configuration...")
    lora_config = LoraConfig(
        task_type=config["lora"]["task_type"],
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
    )

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


# Maintain module-level state for the reward function (avoids adding attributes to the function itself)
@dataclass
class RewardFunctionState:
    call_count: int = 0
    last_game_stats: Optional[Dict[str, Any]] = None


reward_state = RewardFunctionState()


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
    # Refresh opponent model every N steps
    global opponent_model
    reward_state.call_count += 1

    # Check if we need to refresh opponent (every opponent_refresh_steps calls)
    if (
        reward_state.call_count > 1
        and reward_state.call_count % opponent_refresh_steps == 0
    ):
        print(f"🔄 Refreshing opponent model (call {reward_state.call_count})...")
        opponent_model = copy.deepcopy(main_model)
        for param in opponent_model.parameters():
            param.requires_grad = False
        opponent_model.eval()
        print("✅ Opponent model refreshed with latest weights")

    # ------------------------------------------------------------------
    # TIMING: Start comprehensive reward function timing
    # ------------------------------------------------------------------
    import time

    reward_start_time = time.time()

    print(
        f"🎮 Playing {len(completions)} prisoner's dilemma games for reward calculation..."
    )

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

    # ------------------------------------------------------------------
    # Phase 1: Generate opponent strategies (GPU-bound, batch optimized)
    # ------------------------------------------------------------------
    generation_start_time = time.time()
    print(f"🧠 Generating opponent strategies for {len(completions)} games...")

    # Try batch generation first (much faster)
    use_batch_generation = len(completions) > 1
    opponent_completions = []

    if use_batch_generation:
        try:
            batch_start = time.time()

            # Build all opponent prompts for batch processing
            opponent_prompts = []
            for i in range(len(completions)):
                prompt = game_env.get_player_prompt(1, "player")  # Player 2 prompt
                opponent_prompts.append(prompt)

            # Batch tokenize all prompts
            inputs = tokenizer(
                opponent_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
            )
            if device.type == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Prepare generation arguments
            generation_kwargs = {
                "max_new_tokens": config["generation"]["max_new_tokens"],
                "temperature": config["generation"]["temperature"],
                "top_p": config["generation"]["top_p"],
                "do_sample": config["generation"]["do_sample"],
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            # Only add top_k if it's specified and positive
            if "top_k" in config["generation"] and config["generation"]["top_k"] > 0:
                generation_kwargs["top_k"] = config["generation"]["top_k"]

            # Batch generate all opponent strategies in single GPU call (HUGE SPEEDUP!)
            with torch.no_grad():
                outputs = opponent_model.generate(**inputs, **generation_kwargs)

            # Decode all opponent completions
            # Fix: Use attention mask to get actual input lengths for each sequence
            input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

            for i, output in enumerate(outputs):
                start = input_lengths[i]  # Actual length of this specific prompt
                opponent_text = tokenizer.decode(
                    output[start:], skip_special_tokens=True
                ).strip()
                opponent_completions.append(opponent_text)

            batch_time = time.time() - batch_start
            avg_per_game = batch_time / len(completions)
            print(
                f"🚀 BATCH OPPONENT GENERATION SUCCESS: {len(completions)} strategies in {batch_time:.3f}s ({avg_per_game:.3f}s per game)"
            )
            use_batch_generation = True  # Success, log this

        except Exception as e:
            batch_time = time.time() - batch_start
            print(f"❌ BATCH OPPONENT GENERATION FAILED after {batch_time:.3f}s: {e}")
            print("🔄 Falling back to individual generation...")
            use_batch_generation = False  # Failed, fall back

    if not use_batch_generation:
        # Fallback to individual generation
        individual_start = time.time()
        print("🔄 Using individual opponent strategy generation")

        for i in range(len(completions)):
            # Generate opponent strategy with frozen model
            prompt = game_env.get_player_prompt(1, "player")  # Player 2 prompt

            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024
            )
            if device.type == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            generation_kwargs = {
                "max_new_tokens": config["generation"]["max_new_tokens"],
                "temperature": config["generation"]["temperature"],
                "top_p": config["generation"]["top_p"],
                "do_sample": config["generation"]["do_sample"],
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            # Only add top_k if it's specified and positive
            if "top_k" in config["generation"] and config["generation"]["top_k"] > 0:
                generation_kwargs["top_k"] = config["generation"]["top_k"]

            with torch.no_grad():
                outputs = opponent_model.generate(**inputs, **generation_kwargs)

            opponent_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            opponent_completions.append(opponent_text)

        individual_time = time.time() - individual_start
        avg_per_game = individual_time / len(completions)
        print(
            f"🔄 INDIVIDUAL OPPONENT GENERATION COMPLETE: {len(completions)} strategies in {individual_time:.3f}s ({avg_per_game:.3f}s per game)"
        )

    generation_time = time.time() - generation_start_time
    print(f"⚡ OPPONENT GENERATION PHASE COMPLETE: {generation_time:.3f}s total")

    # ------------------------------------------------------------------
    # Phase 2: Game simulation setup and execution (CPU-bound, parallel)
    # ------------------------------------------------------------------
    simulation_setup_start_time = time.time()

    # Build strategy pairs with proper code extraction and validation
    strategy_pairs = []
    for i, (main_completion, opp_completion) in enumerate(
        zip(completions, opponent_completions)
    ):
        # Extract and validate player 1 strategy (main model)
        p1_extracted_code = game_env.extract_code_from_response(main_completion)
        p1_is_valid, p1_error_msg = game_env.validate_submission(
            p1_extracted_code, 0, "player"
        )

        player1_sub = PlayerSubmission(
            player_id=0,
            role="player",
            prompt=game_env.get_player_prompt(0, "player"),
            response=main_completion,
            extracted_code=p1_extracted_code,
            compilation_success=p1_is_valid,
            compilation_error=p1_error_msg if not p1_is_valid else None,
        )

        # Extract and validate player 2 strategy (opponent model)
        p2_extracted_code = game_env.extract_code_from_response(opp_completion)
        p2_is_valid, p2_error_msg = game_env.validate_submission(
            p2_extracted_code, 1, "player"
        )

        player2_sub = PlayerSubmission(
            player_id=1,
            role="player",
            prompt=game_env.get_player_prompt(1, "player"),
            response=opp_completion,
            extracted_code=p2_extracted_code,
            compilation_success=p2_is_valid,
            compilation_error=p2_error_msg if not p2_is_valid else None,
        )

        # Create a fresh copy of game_env for each game to avoid threading issues
        game_env_copy = copy.deepcopy(game_env)
        strategy_pairs.append((player1_sub, player2_sub, game_env_copy, i))

    simulation_setup_time = time.time() - simulation_setup_start_time

    # Simulate games (potentially in parallel)
    parallel_games = config["training"].get("parallel_games", True)
    num_workers = config["training"].get("num_workers", None)

    simulation_start_time = time.time()
    if parallel_games and len(strategy_pairs) > 1:
        max_workers = num_workers or (os.cpu_count() or 1)
        print(
            f"⚙️  Running {len(strategy_pairs)} games in parallel with {max_workers} workers"
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            trajectories = list(executor.map(simulate_single_game, strategy_pairs))
    else:
        print(f"🔄 Running {len(strategy_pairs)} games sequentially")
        trajectories = [simulate_single_game(pair) for pair in strategy_pairs]

    simulation_time = time.time() - simulation_start_time
    avg_simulation_per_game = simulation_time / len(strategy_pairs)
    print(
        f"⚡ GAME SIMULATION PHASE COMPLETE: {simulation_time:.3f}s total ({avg_simulation_per_game:.3f}s per game)"
    )

    # Calculate rewards
    for trajectory in trajectories:
        # Check individual compilation success for each player
        p1_compilation_success = trajectory.player1_submission.compilation_success
        p2_compilation_success = trajectory.player2_submission.compilation_success

        # Determine reward based on compilation success and game outcome
        if p1_compilation_success and p2_compilation_success:
            # Both players have valid code - play the game normally
            game_stats["successful_games"] += 1

            # Extract player rewards from game result
            p1_reward = trajectory.game_result.player_rewards.get(0, 0)
            p2_reward = trajectory.game_result.player_rewards.get(1, 0)

            # Debug: Check if rewards are unexpectedly -1.0
            if p1_reward == -1.0 and p2_reward == -1.0:
                print(
                    f"🚨 DEBUG: Both players got -1.0 despite successful_submissions={trajectory.game_result.successful_submissions}"
                )
                print(
                    f"   Player rewards dict: {trajectory.game_result.player_rewards}"
                )
                print(f"   Game data: {trajectory.game_result.game_data}")
                if hasattr(trajectory.game_result, "execution_logs"):
                    print(
                        f"   Execution logs: {trajectory.game_result.execution_logs[-5:]}"
                    )  # Last 5 logs

            # Determine winner and assign GRPO reward (for main model)
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

        elif not p1_compilation_success and not p2_compilation_success:
            # Both players have invalid code - both get penalized
            reward = -1.0  # Main model gets penalized for its own failure
            game_stats["execution_failures"] += 1
            game_stats["both_players_failed"] = (
                game_stats.get("both_players_failed", 0) + 1
            )

        elif not p1_compilation_success:
            # Only main model (Player 1) has invalid code
            reward = -1.0  # Main model gets penalized for its own failure
            game_stats["execution_failures"] += 1
            game_stats["player1_failed"] = game_stats.get("player1_failed", 0) + 1

        else:  # not p2_compilation_success
            # Only opponent (Player 2) has invalid code
            reward = 0.0  # Main model gets neutral reward, opponent gets penalized
            game_stats["execution_failures"] += 1
            game_stats["player2_failed"] = game_stats.get("player2_failed", 0) + 1

        rewards.append(reward)

    # ------------------------------------------------------------------
    # Phase 3: Statistics calculation and comprehensive timing
    # ------------------------------------------------------------------
    statistics_start_time = time.time()

    # Store stats for logging with comprehensive timing data
    total_reward_time = time.time() - reward_start_time
    statistics_time = time.time() - statistics_start_time
    other_time = (
        total_reward_time
        - generation_time
        - simulation_setup_time
        - simulation_time
        - statistics_time
    )

    # Add timing information to game stats
    game_stats.update(
        {
            # Comprehensive timing breakdown
            "reward_timing/total_time": total_reward_time,
            "reward_timing/generation_time": generation_time,
            "reward_timing/simulation_setup_time": simulation_setup_time,
            "reward_timing/simulation_time": simulation_time,
            "reward_timing/statistics_time": statistics_time,
            "reward_timing/other_time": other_time,
            "reward_timing/batch_generation_used": use_batch_generation,
            "reward_timing/parallel_simulation_used": parallel_games
            and len(strategy_pairs) > 1,
            "reward_timing/num_workers": (
                max_workers if parallel_games and len(strategy_pairs) > 1 else 1
            ),
            "reward_timing/avg_generation_per_game": generation_time / len(completions),
            "reward_timing/avg_simulation_per_game": avg_simulation_per_game,
        }
    )

    reward_state.last_game_stats = game_stats

    # Print comprehensive timing breakdown
    print(f"⏱️  REWARD FUNCTION TIMING BREAKDOWN:")
    print(f"   Total: {total_reward_time:.3f}s")
    print(
        f"   Generation: {generation_time:.3f}s ({100*generation_time/total_reward_time:.1f}%)"
    )
    print(
        f"   Simulation: {simulation_time:.3f}s ({100*simulation_time/total_reward_time:.1f}%)"
    )
    print(f"   Other: {other_time:.3f}s ({100*other_time/total_reward_time:.1f}%)")

    # Performance optimizations summary
    optimization_summary = []
    if use_batch_generation:
        optimization_summary.append("🚀 Batch Generation")
    else:
        optimization_summary.append("🔄 Individual Generation")

    if parallel_games and len(strategy_pairs) > 1:
        optimization_summary.append(f"⚙️  Parallel Simulation ({max_workers} workers)")
    else:
        optimization_summary.append("🔄 Sequential Simulation")

    print(f"⚡ Optimizations: {' | '.join(optimization_summary)}")

    # Show some debug info
    debug_config = config.get("debug", {})
    show_detailed_games = debug_config.get("show_detailed_games", 0)

    if show_detailed_games > 0:
        print(
            f"🎯 Showing details for first {min(show_detailed_games, len(trajectories))} games:"
        )
        for i in range(min(show_detailed_games, len(trajectories))):
            trajectory = trajectories[i]
            reward = rewards[i]
            print(f"\n🎮 Game {i+1}: Reward = {reward:+.1f}")

            # Check compilation status
            p1_compilation = trajectory.player1_submission.compilation_success
            p2_compilation = trajectory.player2_submission.compilation_success

            if p1_compilation and p2_compilation:
                # Both players have valid code
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
                # Show compilation failure details
                if not p1_compilation and not p2_compilation:
                    print(f"   ❌ Both players failed compilation")
                elif not p1_compilation:
                    print(f"   ❌ Player 1 (main model) failed compilation")
                else:
                    print(f"   ❌ Player 2 (opponent) failed compilation")

                # Show the actual generated strategies to debug
                p1_code = trajectory.player1_submission.extracted_code[:200]
                p2_code = trajectory.player2_submission.extracted_code[:200]
                print(f"   P1 strategy (first 200 chars): {p1_code}")
                print(f"   P2 strategy (first 200 chars): {p2_code}")

                # Show compilation errors
                if not p1_compilation:
                    print(
                        f"   P1 compilation error: {trajectory.player1_submission.compilation_error}"
                    )
                if not p2_compilation:
                    print(
                        f"   P2 compilation error: {trajectory.player2_submission.compilation_error}"
                    )

                # Show execution logs if available
                if (
                    hasattr(trajectory.game_result, "execution_logs")
                    and trajectory.game_result.execution_logs
                ):
                    print(
                        f"   Execution logs: {trajectory.game_result.execution_logs[-3:]}"
                    )  # Last 3 logs

    print(
        f"🏆 Game results: {game_stats['player1_wins']} wins, {game_stats['player2_wins']} losses, {game_stats['ties']} ties"
    )
    print(
        f"💥 Execution failures: {game_stats['execution_failures']}/{len(completions)}"
    )

    # Show detailed failure breakdown if any failures occurred
    if game_stats["execution_failures"] > 0:
        p1_failures = game_stats.get("player1_failed", 0)
        p2_failures = game_stats.get("player2_failed", 0)
        both_failures = game_stats.get("both_players_failed", 0)

        failure_details = []
        if p1_failures > 0:
            failure_details.append(f"P1 failed: {p1_failures}")
        if p2_failures > 0:
            failure_details.append(f"P2 failed: {p2_failures}")
        if both_failures > 0:
            failure_details.append(f"Both failed: {both_failures}")

        if failure_details:
            print(f"   📊 Failure breakdown: {', '.join(failure_details)}")

    # Print both players' strategies for one game per iteration
    if len(trajectories) > 0:
        print(f"\n{'='*80}")
        print(
            f"🎮 BOTH PLAYERS' STRATEGIES FOR GAME 1 (Step {reward_state.call_count})"
        )
        print(f"{'='*80}")

        # Get first game details
        trajectory = trajectories[0]
        reward = rewards[0]

        print(f"🎯 Game Reward: {reward:+.2f}")

        # Check compilation status for detailed reporting
        p1_compilation = trajectory.player1_submission.compilation_success
        p2_compilation = trajectory.player2_submission.compilation_success

        if p1_compilation and p2_compilation:
            print(f"🔧 Game Success: ✅ Both players compiled successfully")
            p1_reward = trajectory.game_result.player_rewards.get(0, 0)
            p2_reward = trajectory.game_result.player_rewards.get(1, 0)
            print(f"📊 Player Scores: P1={p1_reward}, P2={p2_reward}")
        elif not p1_compilation and not p2_compilation:
            print(f"🔧 Game Success: ❌ Both players failed compilation")
        elif not p1_compilation:
            print(f"🔧 Game Success: ❌ Player 1 (main model) failed compilation")
        else:
            print(f"🔧 Game Success: ❌ Player 2 (opponent) failed compilation")

        # Show Player 1 strategy (main model)
        p1_code = trajectory.player1_submission.extracted_code
        p1_compilation = (
            "✅ Valid"
            if trajectory.player1_submission.compilation_success
            else "❌ Invalid"
        )
        print(f"\n🤖 PLAYER 1 (Main Model) Strategy:")
        print(f"🔧 Compilation: {p1_compilation}")
        if not trajectory.player1_submission.compilation_success:
            print(f"❌ Error: {trajectory.player1_submission.compilation_error}")
        print(f"```python\n{p1_code}\n```")

        # Show Player 2 strategy (opponent model)
        p2_code = trajectory.player2_submission.extracted_code
        p2_compilation = (
            "✅ Valid"
            if trajectory.player2_submission.compilation_success
            else "❌ Invalid"
        )
        print(f"\n🥶 PLAYER 2 (Frozen Opponent) Strategy:")
        print(f"🔧 Compilation: {p2_compilation}")
        if not trajectory.player2_submission.compilation_success:
            print(f"❌ Error: {trajectory.player2_submission.compilation_error}")
        print(f"```python\n{p2_code}\n```")

        # Show game features if any
        game_data = trajectory.game_result.game_data
        features = []
        if game_data.get("wsls_bot_used", False):
            features.append("🤖 WSLS Bot")
        if game_data.get("noise_rounds", 0) > 0:
            features.append(f"🎲 Noise ({game_data['noise_rounds']} rounds)")
        if features:
            print(f"\n🎮 Game Features: {', '.join(features)}")

        # Show execution logs if available
        if (
            hasattr(trajectory.game_result, "execution_logs")
            and trajectory.game_result.execution_logs
        ):
            print(f"\n📝 Execution Logs (last 5):")
            for log in trajectory.game_result.execution_logs[-5:]:
                print(f"   {log}")

        print(f"{'='*80}")

    return rewards


# Adaptive settings based on GPU
training_args = config["grpo_config"]
if platform_info["gpu_type"] == "V100":
    training_args["per_device_train_batch_size"] = min(
        training_args["per_device_train_batch_size"], 2
    )
    training_args["gradient_accumulation_steps"] = max(
        training_args.get("gradient_accumulation_steps", 1), 4
    )
    print("🔧 Adjusted training args for V100 memory constraints")

# Override GRPO max_steps with training.num_steps
training_args["max_steps"] = config["training"]["num_steps"]
print(
    f"🔧 Set GRPO max_steps to {config['training']['num_steps']} (from training.num_steps)"
)

# Fix model config path for GRPOTrainer if in offline mode
if offline_mode:
    # Point the model config to actual cached snapshot directory so GRPOTrainer can find tokenizer files locally
    cached_model_dirs = glob.glob(
        os.path.join(
            cache_dir,
            f"models--{model_id.replace('/', '--')}",
            "snapshots",
            "*",
        )
    )
    if cached_model_dirs:
        # Point config to local snapshot; ignore typing since field is internal/private
        main_model.config._name_or_path = cached_model_dirs[0]  # type: ignore[attr-defined, assignment]
        print(f"🔧 Set model config path for offline mode: {cached_model_dirs[0]}")
    else:
        print("⚠️ Warning: Could not find cached model directory for offline mode")

# Create GRPO config
grpo_config = GRPOConfig(**training_args)

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
        run_name = getattr(wandb.run, "name", None)
        print(f"✅ Initialized W&B run: {run_name} (Project: {project_name})")
    else:
        # Offline mode: Use consistent project name without timestamp
        project_name = config["wandb"]["project_name_prefix"]
        wandb.init(
            project=project_name,
            config={**config, **seed_manager.get_seed_info()},
            mode="offline",
        )
        run_name = getattr(wandb.run, "name", None)
        print(f"✅ Initialized W&B run (offline): {run_name} (Project: {project_name})")

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

# Run initial alignment evaluation if enabled
if alignment_evaluator.config.enabled and alignment_evaluator.config.eval_at_start:
    print("🧪 Running initial alignment evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("initial_alignment")
    initial_alignment_results = alignment_evaluator.evaluate_model(
        main_model, tokenizer, step=0, phase="initial"
    )

    if WANDB_ENABLED and wandb.run:
        alignment_log_dict = alignment_evaluator.get_wandb_log_dict(initial_alignment_results)
        alignment_log_dict["step"] = 0
        wandb.log(alignment_log_dict)

# Add interval evaluation callback
from transformers import TrainerCallback


class IntervalEvaluationCallback(TrainerCallback):
    def __init__(
        self, evaluator, model, tokenizer, config, wandb_enabled, seed_manager, alignment_evaluator=None
    ):
        self.evaluator = evaluator
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.wandb_enabled = wandb_enabled
        self.seed_manager = seed_manager
        self.alignment_evaluator = alignment_evaluator
        self.eval_interval = config["evaluation"].get("eval_interval_steps", None)
        self.alignment_eval_interval = config.get("alignment_evaluation", {}).get("eval_interval_steps", None)

    def on_step_end(self, args, state, control, **kwargs):
        # Log reward function statistics if available
        if (
            self.wandb_enabled
            and wandb.run
            and reward_state.last_game_stats is not None
        ):
            # Log game statistics and timing from reward function
            game_stats = reward_state.last_game_stats
            log_dict = {
                f"game/{key}": value
                for key, value in game_stats.items()
                if isinstance(value, (int, float, bool))
            }
            log_dict["step"] = state.global_step
            wandb.log(log_dict)

        # Run interval MBPP evaluation if configured
        if (
            self.config["evaluation"].get("enabled_interval", False)
            and self.evaluator.config.enabled
            and self.eval_interval
            and state.global_step > 0
            and state.global_step % self.eval_interval == 0
        ):
            print(f"🧪 Running interval MBPP evaluation at step {state.global_step}...")
            # Seed for consistent evaluation
            self.seed_manager.seed_for_evaluation_auto(
                f"interval_step_{state.global_step}"
            )
            interval_results = self.evaluator.evaluate_model(
                self.model, self.tokenizer, step=state.global_step, phase="interval"
            )

            if self.wandb_enabled and wandb.run and "pass_rate" in interval_results:
                wandb.log(
                    {
                        "mbpp_eval/pass_rate": interval_results["pass_rate"],
                        "mbpp_eval/problems_passed": interval_results[
                            "problems_passed"
                        ],
                        "mbpp_eval/total_problems": interval_results["total_problems"],
                        "mbpp_eval/eval_time": interval_results["eval_time_seconds"],
                        "step": state.global_step,
                    }
                )

        # Run interval alignment evaluation if configured
        if (
            self.alignment_evaluator
            and self.alignment_evaluator.config.enabled
            and self.config.get("alignment_evaluation", {}).get("enabled_interval", False)
            and self.alignment_eval_interval
            and state.global_step > 0
            and state.global_step % self.alignment_eval_interval == 0
        ):
            print(f"🧪 Running interval alignment evaluation at step {state.global_step}...")
            # Seed for consistent evaluation
            self.seed_manager.seed_for_evaluation_auto(
                f"interval_alignment_step_{state.global_step}"
            )
            alignment_results = self.alignment_evaluator.evaluate_model(
                self.model, self.tokenizer, step=state.global_step, phase="interval"
            )

            if self.wandb_enabled and wandb.run:
                alignment_log_dict = self.alignment_evaluator.get_wandb_log_dict(alignment_results)
                alignment_log_dict["step"] = state.global_step
                wandb.log(alignment_log_dict)


# Add the callback to trainer
interval_callback = IntervalEvaluationCallback(
    mbpp_evaluator,
    main_model,
    tokenizer,
    config,
    WANDB_ENABLED,
    seed_manager,
    alignment_evaluator,
)
grpo_trainer.add_callback(interval_callback)

print("🎮 Starting GRPO Prisoner's Dilemma Training")

# Run GRPO training - this will automatically call our reward function
print("🏋️ Starting GRPO training with prisoner's dilemma rewards...")
grpo_trainer.train()

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

# Run final alignment evaluation if enabled
if alignment_evaluator.config.enabled and alignment_evaluator.config.eval_at_end:
    print("🧪 Running final alignment evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("final_alignment")
    final_alignment_results = alignment_evaluator.evaluate_model(
        main_model, tokenizer, step=num_steps, phase="final"
    )

    if WANDB_ENABLED and wandb.run:
        final_alignment_log_dict = alignment_evaluator.get_wandb_log_dict(final_alignment_results)
        final_alignment_log_dict["step"] = num_steps
        wandb.log(final_alignment_log_dict)

# Save final model
final_checkpoint_dir = f"{config['training']['checkpoint_dir']}/final"
print(f"💾 Saving final model...")
main_model.save_pretrained(final_checkpoint_dir)
tokenizer.save_pretrained(final_checkpoint_dir)
print(f"✅ Final model saved to {final_checkpoint_dir}")

if WANDB_ENABLED and wandb.run:
    wandb.finish()

print("🎉 Training completed successfully!")
