#!/usr/bin/env python3
"""
GRPO Code Generation Game - Qwen vs Static Opponent
Based on the working RFT One Way example
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
from utils.seed_manager import SeedManager
from evaluation import (
    MBPPEvaluator,
    create_eval_config_for_training,
    print_config_summary,
)

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

# Initialize comprehensive seed management
print("🎲 Setting up seed management...")
# Create a basic config for seed manager since this script doesn't use config files
# All settings can be overridden by environment variables
seed_config = {
    "seed": int(os.environ.get("SEED", "42")),  # Default seed, override with SEED env var
    "evaluation": {
        "consistent_questions": os.environ.get("EVAL_CONSISTENT_QUESTIONS", "true").lower() in ("true", "1", "yes", "on")
    }
}
seed_manager = SeedManager.from_config(seed_config)
seed_manager.seed_everything()

# Initialize MBPP evaluator with configurable settings
eval_config = create_eval_config_for_training("grpo_code_game")
print_config_summary(eval_config)

mbpp_evaluator = MBPPEvaluator(eval_config)

if not mbpp_evaluator.config.enabled:
    print("⚠️ MBPP evaluation disabled - dataset not found")
    print("💡 To enable evaluation, download MBPP dataset first:")
    print("   python -m evaluation.mbpp_evaluator")
else:
    print(
        f"✅ MBPP evaluation enabled with {mbpp_evaluator.config.num_questions} questions"
    )


def run_and_capture(code: str) -> str:
    """Executes code and captures its stdout output."""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {})  # empty global scope
    except Exception as e:
        return f"Execution error: {e}"
    return buffer.getvalue().strip()


# Define the formatted prompt
prompt = (
    "Write a Python program that is difficult for another model trained on SmollM-135M-Instruct data to predict. "
    "The program should return a singular integer value. "
    "Then, show only the exact output of running your program.\n\n"
    "Format your response exactly like these examples:\n\n"
    "```python\n"
    "def tricky():\n"
    "    return int('0b1011', 2)\n"
    "print(tricky())\n"
    "```\n"
    "```output\n"
    "11\n"
    "```\n\n"
    "```python\n"
    "def f():\n"
    "    return sum([i % 3 for i in range(10)])\n"
    "print(f())\n"
    "```\n"
    "```output\n"
    "10\n"
    "```\n\n"
    "Now you try:\n"
)

# Create a dataset of 1000 identical prompts
dataset = Dataset.from_dict({"prompt": [prompt] * 1000})

print(f"Created dataset: {dataset}")

# Set up model cache directory
cache_dir = "./model_cache"
os.makedirs(cache_dir, exist_ok=True)

# Use default model (or from environment variable)
model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B")
print(f"📥 Using model: {model_id}")

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

# Load static opponent model - use same cached model as trainable model
model_name2 = model_id  # Use same model as the trainable model
print(f"Loading static opponent: {model_name2}")

tokenizer2 = AutoTokenizer.from_pretrained(
    model_name2, cache_dir=cache_dir, local_files_only=offline_mode
)
# Ensure tokenizer has pad token
if tokenizer2.pad_token is None:
    tokenizer2.pad_token = tokenizer2.eos_token

model2 = AutoModelForCausalLM.from_pretrained(
    model_name2, torch_dtype="auto", cache_dir=cache_dir, local_files_only=offline_mode
)

# Move to GPU if available
model2 = model2.to("cuda" if torch.cuda.is_available() else "cpu")

# Create text generation pipeline for opponent
generator2 = pipeline("text-generation", model=model2, tokenizer=tokenizer2)


def reward_function(completions, **kwargs):
    """Reward function for GRPO training."""
    rewards = []

    successful_predictions = (
        0  # Trainable model "wins" (opponent fails to predict correctly)
    )
    failed_predictions = 0  # Trainable model "losses" (opponent predicts correctly)

    for i, comp in enumerate(completions):
        if not isinstance(comp, str):
            rewards.append(-1)  # invalid completion
            failed_predictions += 1
            continue

        # Extract code according to schema
        code = re.search(r"```python\s*\n(.*?)```", comp, re.DOTALL)
        if code:
            code = code.group(1).strip()
        else:
            code = ""

        expected_output = re.search(r"```output\s*(.*?)```", comp, re.DOTALL)
        if expected_output:
            expected_output = expected_output.group(1).strip()
        else:
            expected_output = ""

        # Get opponent prediction
        prompt2 = (
            "Examine this code and predict the integer output.\n"
            f"{code}\n\n"
            "Do not include any text, markdown, or explanation, just the number."
        )

        model_pred = ""  # Initialize model_pred to ensure it's always defined
        try:
            # Seed for deterministic opponent predictions
            seed_manager.seed_for_generation(step=i, generation_idx=0)
            model_pred_raw = generator2(
                prompt2, max_new_tokens=200, do_sample=True, temperature=0.7
            )[0]["generated_text"]

            # Extract first number from prediction
            model_pred_search = re.search(r"\b(-?\d+)\b", model_pred_raw)
            model_pred = model_pred_search.group(1) if model_pred_search else ""
            try:
                model_pred = int(model_pred)
            except ValueError:
                model_pred = "ERROR: Conversion to integer failed"
        except Exception as e:
            model_pred = f"ERROR: {e}"

        # Get true output
        true_output = run_and_capture(code)

        # Debug output (show first few)
        if len(rewards) < 3:
            print("=" * 50)
            print(f"Model prediction: {model_pred}")
            print(f"Code: {code}")
            print(f"Expected output: {expected_output}")
            print(f"True output: {true_output}")
            print(f"Original completion: {comp[:200]}...")
            print("=" * 50)

        # Calculate reward: +1 if opponent is wrong, -1 if opponent is right
        reward = -1  # Default to error/loss
        try:
            # Convert both to string for robust comparison, as model_pred can be int or error string
            if str(model_pred) == str(true_output).strip():
                reward = -1  # Opponent guessed correctly
                failed_predictions += 1
            else:
                reward = 1  # Opponent guessed incorrectly (good for us)
                successful_predictions += 1
        except Exception as e:
            print(f"Error comparing outputs in reward_function: {e}")
            reward = -1  # Error case, treat as a loss for the trainable model
            failed_predictions += 1

        # Show progress for all completions
        print(
            f"Completion {i+1}/{len(completions)}: pred={model_pred}, true={true_output}, reward={reward}"
        )

        rewards.append(reward)

    # Log metrics to wandb after processing the entire batch of completions
    if wandb.run:  # Ensure wandb is initialized before logging
        avg_batch_reward = sum(rewards) / len(rewards) if rewards else 0

        wandb.log(
            {
                "reward/avg_batch_reward": avg_batch_reward,
                "reward/successful_predictions_count": successful_predictions,
                "reward/failed_predictions_count": failed_predictions,
                "reward/total_completions_processed": len(completions),
            }
        )

    return rewards


# Training arguments - adjust for GPU memory capacity and model size
print("Setting up GRPO training...")

# Adaptive settings based on GPU type and model size
if platform_info["gpu_type"] == "V100":
    if "3B" in model_id:
        # V100 with 3B model - very aggressive memory reduction
        batch_size = 1
        gradient_accumulation_steps = 8
        max_prompt_length = 128
        max_completion_length = 64
        print("🔧 Using V100 + 3B model settings (very aggressive memory reduction)")
    else:
        # V100 with 1.5B model - moderate memory reduction
        batch_size = 4
        gradient_accumulation_steps = 2
        max_prompt_length = 256
        max_completion_length = 128
        print("🔧 Using V100 + 1.5B model settings (moderate memory reduction)")
else:
    # A100 or other GPUs - standard settings
    batch_size = 8
    gradient_accumulation_steps = 2
    max_prompt_length = 512
    max_completion_length = 200
    print("🔧 Using standard memory settings for A100/other GPUs")

training_args = GRPOConfig(
    output_dir="checkpoints/grpo_code",
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
    reward_funcs=[reward_function],
    args=training_args,
    train_dataset=dataset,
)

print("Starting GRPO training...")

# Initialize wandb run (only if not in offline mode)
if not offline_mode:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb.init(project=f"qwen-code-game-grpo-{timestamp}", config=seed_manager.get_seed_info())
else:
    print("🚫 Skipping wandb initialization (offline mode)")

# Run initial evaluation if enabled
if mbpp_evaluator.should_evaluate(is_start=True):
    print("🧪 Running initial MBPP evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("initial")
    initial_results = mbpp_evaluator.evaluate_model(
        model1, tokenizer1, step=0, phase="initial"
    )

    if wandb.run and initial_results.get("enabled", False):
        wandb.log(
            {
                "mbpp_eval/initial_pass_rate": initial_results["pass_rate"],
                "mbpp_eval/initial_problems_passed": initial_results["problems_passed"],
                "mbpp_eval/initial_total_problems": initial_results["total_problems"],
                "mbpp_eval/initial_eval_time": initial_results["eval_time_seconds"],
            }
        )

# Train model
trainer.train()

# Run final evaluation if enabled
if mbpp_evaluator.should_evaluate(is_end=True):
    print("🧪 Running final MBPP evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("final")
    final_results = mbpp_evaluator.evaluate_model(
        model1, tokenizer1, step=trainer.state.global_step, phase="final"
    )

    if wandb.run and final_results.get("enabled", False):
        wandb.log(
            {
                "mbpp_eval/final_pass_rate": final_results["pass_rate"],
                "mbpp_eval/final_problems_passed": final_results["problems_passed"],
                "mbpp_eval/final_total_problems": final_results["total_problems"],
                "mbpp_eval/final_eval_time": final_results["eval_time_seconds"],
            }
        )

print("Training completed!")
if not offline_mode:
    wandb.finish()
else:
    print("✅ Training completed (offline mode - no wandb to finish)")
