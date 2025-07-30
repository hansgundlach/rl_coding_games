#!/usr/bin/env python3
"""
GRPO Code Execution Training - Model learns to generate executable code
This environment rewards the model for generating Python code that runs successfully.
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
import subprocess
import tempfile
import time
import signal
import yaml
import argparse


print("🚀 Starting GRPO Code Execution Training...")
print("📋 Initializing components...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="GRPO Code Execution Training")
parser.add_argument(
    "--config",
    type=str,
    default="configs/grpo_code_execution.yaml",
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
        os.environ.pop(
            "WANDB_MODE", None
        )  # Remove explicit WANDB_MODE if it was set to disabled
else:  # If WANDB_ENABLED is False, explicitly disable W&B
    os.environ["WANDB_MODE"] = "disabled"
    print("🚫 W&B logging explicitly disabled by user configuration.")

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("📦 Loading utility modules...")
from utils.env_loader import get_api_key
from utils.vllm_client import (
    initialize_vllm_integration,
    cleanup_vllm_integration,
    get_vllm_integration,
)
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
eval_config_dict = config.get('evaluation', {})

# Update results directory to logs folder if running in SLURM
log_dir = os.environ.get('GRPO_LOG_DIR', 'logs')
if 'GRPO_LOG_DIR' in os.environ:
    eval_config_dict['results_dir'] = log_dir
    print(f"📊 Evaluation results will be saved to: {log_dir}")

# Create EvalConfig object from consolidated config
eval_config = EvalConfig(**eval_config_dict)

# Print evaluation config summary
print("\n" + "="*50)
print("📊 MBPP Evaluation Configuration")
print("="*50)
print(f"Enabled: {'✅' if eval_config.enabled else '❌'}")
if eval_config.enabled:
    print(f"Questions: {eval_config.num_questions}")
    print(f"Initial eval: {'✅' if config['evaluation']['enabled_initial'] else '❌'}")
    print(f"Final eval: {'✅' if config['evaluation']['enabled_final'] else '❌'}")
    print(f"Dataset: {eval_config.dataset_path or 'auto-detect'}")
    print(f"Results dir: {eval_config.results_dir}")
    print(f"Temperature: {eval_config.temperature}")
    print(f"Max tokens: {eval_config.max_new_tokens}")
    print(f"Timeout: {eval_config.timeout_seconds}s")
print("="*50 + "\n")

mbpp_evaluator = MBPPEvaluator(eval_config)

if not mbpp_evaluator.config.enabled:
    print("⚠️ MBPP evaluation disabled - dataset not found")
    print("💡 To enable evaluation, download MBPP dataset first:")
    print("   python -m evaluation.mbpp_evaluator")
else:
    print(
        f"✅ MBPP evaluation enabled with {mbpp_evaluator.config.num_questions} questions"
    )


def safe_execute_code(code: str, timeout: int = 5) -> dict:
    """
    Safely execute Python code in a sandboxed environment.

    Returns:
        dict: {
            'success': bool,
            'output': str,
            'error': str,
            'execution_time': float,
            'timeout': bool
        }
    """
    result = {
        "success": False,
        "output": "",
        "error": "",
        "execution_time": 0.0,
        "timeout": False,
    }

    # Create a temporary file with the code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        start_time = time.time()

        # Execute code in subprocess with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tempfile.gettempdir(),  # Run in temp directory
            env={"PYTHONPATH": ""},  # Minimal environment
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            result["execution_time"] = time.time() - start_time
            result["output"] = stdout.strip()
            result["error"] = stderr.strip()
            result["success"] = process.returncode == 0 and not stderr

        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()  # Clean up
            result["timeout"] = True
            result["error"] = f"Code execution timed out after {timeout} seconds"
            result["execution_time"] = timeout

    except Exception as e:
        result["error"] = f"Execution error: {str(e)}"
        result["execution_time"] = time.time() - start_time

    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except:
            pass

    return result


# Define the code generation prompt
prompt = (
    "Write a complete Python program that demonstrates a programming concept or solves a simple problem. "
    "The program should be executable and produce some output. "
    "Focus on writing clean, working code rather than complex algorithms.\n\n"
    "Examples of good programs:\n"
    "- Mathematical calculations\n"
    "- String manipulations\n"
    "- List operations\n"
    "- Simple algorithms\n"
    "- Basic data structures\n\n"
    "Format your response with just the Python code:\n"
    "```python\n"
    "# Your code here\n"
    "```\n\n"
    "Write a Python program:\n"
)

# Create a dataset of prompts
dataset_size = config["dataset"]["size"]
dataset = Dataset.from_dict({"prompt": [prompt] * dataset_size})

print(f"Created dataset: {dataset}")

# Set up model cache directory
cache_dir = config["model"]["cache_dir"]
os.makedirs(cache_dir, exist_ok=True)

# Load the main model (trainable) - prefer 1.5B for better memory efficiency
if offline_mode:
    # Check what models are available in cache, prefer 1.5B over 3B for memory efficiency
    cached_models = glob.glob(os.path.join(cache_dir, "models--Qwen--Qwen2.5-*"))

    # Prefer 1.5B model if available (better memory efficiency)
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
        model_id = "Qwen/Qwen2.5-1.5B"  # fallback
        print(f"⚠️  No cached models found, attempting: {model_id}")
else:
    model_id = config["model"]["id"]  # Use configured model

print(f"📥 Loading trainable model: {model_id}")
print("⏳ This may take 2-3 minutes depending on model size and storage speed...")

model1 = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_dir,
    local_files_only=offline_mode,
)
print("🔤 Loading tokenizer...")
tokenizer1 = AutoTokenizer.from_pretrained(
    model_id, cache_dir=cache_dir, local_files_only=offline_mode
)
# Ensure tokenizer has pad token
if tokenizer1.pad_token is None:
    tokenizer1.pad_token = tokenizer1.eos_token

# Load LoRA
print("🔧 Setting up LoRA configuration...")
lora_config = LoraConfig(
    task_type=config["lora"]["task_type"],
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["lora_alpha"],
    target_modules=config["lora"]["target_modules"],
)
print("🎯 Applying LoRA to model...")
model1 = get_peft_model(model1, lora_config)
print(model1.print_trainable_parameters())

# Initialize vLLM integration if enabled
vllm_integration = None
if config.get("vllm", {}).get("enabled", False):
    print("🚀 Initializing vLLM integration...")
    try:
        vllm_integration = initialize_vllm_integration(
            config["vllm"], model_id, offline_mode
        )
        if vllm_integration.initialize():
            print("✅ vLLM server started successfully")
        else:
            print("⚠️ vLLM server failed to start, will use HuggingFace fallback")
    except Exception as e:
        print(f"⚠️ vLLM initialization failed: {e}")
        print("   Continuing with HuggingFace generation...")
else:
    print("📝 vLLM integration disabled, using HuggingFace generation")


def execution_reward_function(completions, **kwargs):
    """
    Reward function for code execution training.
    Uses vLLM for faster completion generation if available.

    Rewards:
    +1.0: Code executes successfully and produces output
    +0.5: Code executes successfully but produces no output
    -0.5: Code has minor errors (syntax errors, import errors)
    -1.0: Code has major errors (runtime errors, timeout)
    """
    rewards = []

    successful_executions = 0
    failed_executions = 0
    timeout_executions = 0
    syntax_errors = 0

    # Performance tracking
    vllm_used = False
    vllm_integration = get_vllm_integration()
    if (
        vllm_integration
        and vllm_integration.vllm_config.enabled
        and vllm_integration.vllm_config.integration.get(
            "use_for_grpo_completions", False
        )
    ):
        vllm_used = True
        if (
            config.get("vllm", {})
            .get("integration", {})
            .get("log_performance_comparison", False)
        ):
            print(f"🚀 Using vLLM for {len(completions)} completions")

    for i, comp in enumerate(completions):
        if not isinstance(comp, str):
            rewards.append(-1.0)  # Invalid completion
            failed_executions += 1
            continue

        # Extract Python code from completion
        code_match = re.search(r"```python\s*\n(.*?)```", comp, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Try to extract code without markdown formatting
            code = comp.strip()

        if not code:
            rewards.append(-1.0)  # No code generated
            failed_executions += 1
            continue

        # Execute the code safely
        execution_result = safe_execute_code(
            code, timeout=config["execution"]["timeout"]
        )

        # Calculate reward based on execution result
        if execution_result["success"]:
            if execution_result["output"]:
                reward = config["rewards"][
                    "success_with_output"
                ]  # Perfect: runs and produces output
                successful_executions += 1
            else:
                reward = config["rewards"][
                    "success_no_output"
                ]  # Good: runs but no output
                successful_executions += 1
        elif execution_result["timeout"]:
            reward = config["rewards"][
                "timeout_error"
            ]  # Bad: infinite loop or too slow
            timeout_executions += 1
        elif (
            "SyntaxError" in execution_result["error"]
            or "IndentationError" in execution_result["error"]
        ):
            reward = config["rewards"]["syntax_error"]  # Minor: syntax issues
            syntax_errors += 1
        else:
            reward = config["rewards"]["runtime_error"]  # Bad: runtime errors
            failed_executions += 1

        # Debug output (show first few)
        if i < config["execution"]["debug_completions"]:
            print("=" * 50)
            print(f"Completion {i+1}:")
            print(f"Code: {code[:200]}...")
            print(f"Success: {execution_result['success']}")
            print(f"Output: {execution_result['output'][:100]}")
            print(f"Error: {execution_result['error'][:100]}")
            print(f"Execution time: {execution_result['execution_time']:.2f}s")
            print(f"Reward: {reward}")
            print("=" * 50)

        # Show progress for all completions
        status = "✅" if execution_result["success"] else "❌"
        print(
            f"Completion {i+1}/{len(completions)}: {status} reward={reward:.1f}, time={execution_result['execution_time']:.2f}s"
        )

        rewards.append(reward)

    # Log metrics to wandb after processing the entire batch
    if WANDB_ENABLED and wandb.run:  # Only log if W&B is enabled and run is active
        avg_batch_reward = sum(rewards) / len(rewards) if rewards else 0
        success_rate = successful_executions / len(completions) if completions else 0

        wandb_metrics = {
            "execution/avg_batch_reward": avg_batch_reward,
            "execution/success_rate": success_rate,
            "execution/successful_executions": successful_executions,
            "execution/failed_executions": failed_executions,
            "execution/timeout_executions": timeout_executions,
            "execution/syntax_errors": syntax_errors,
            "execution/total_completions": len(completions),
        }

        # Add vLLM performance tracking
        if vllm_used:
            wandb_metrics["vllm/used_for_completions"] = True
            wandb_metrics["vllm/batch_size"] = len(completions)

        wandb.log(wandb_metrics)

    return rewards


# Training arguments - adjust for GPU memory capacity and model size
print("Setting up GRPO training for code execution...")

# Adaptive settings based on GPU type and model size
if platform_info["gpu_type"] == "V100":
    if "3B" in model_id:
        # V100 with 3B model - very aggressive memory reduction
        batch_size = 1
        gradient_accumulation_steps = 8
        max_prompt_length = 128
        max_completion_length = 256  # Longer for code generation
        print("🔧 Using V100 + 3B model settings (very aggressive memory reduction)")
    else:
        # V100 with 1.5B model - moderate memory reduction
        batch_size = 4
        gradient_accumulation_steps = 2
        max_prompt_length = 256
        max_completion_length = 384  # Longer for code generation
        print("🔧 Using V100 + 1.5B model settings (moderate memory reduction)")
else:
    # A100 or other GPUs - standard settings
    batch_size = 8
    gradient_accumulation_steps = 2
    max_prompt_length = 512
    max_completion_length = 512  # Longer for code generation
    print("🔧 Using standard memory settings for A100/other GPUs")

# Update output directory to logs folder if running in SLURM
output_dir = config["training_args"]["output_dir"]
if 'GRPO_LOG_DIR' in os.environ:
    output_dir = os.path.join(os.environ['GRPO_LOG_DIR'], 'checkpoints')
    print(f"💾 Checkpoints will be saved to: {output_dir}")

training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=config["training_args"]["learning_rate"],
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    num_generations=config["training_args"]["num_generations"],
    optim=config["training_args"]["optim"],
    num_train_epochs=config["training_args"]["num_train_epochs"],
    bf16=platform_info["supports_bf16"],  # Auto-configured based on GPU type
    fp16=not platform_info["supports_bf16"],  # Use fp16 if bf16 not supported
    gradient_checkpointing=config["training_args"]["gradient_checkpointing"],
    dataloader_pin_memory=(
        False
        if platform_info["gpu_type"] == "V100"
        else config["training_args"]["dataloader_pin_memory"]
    ),  # Memory optimization for V100
    report_to=["wandb"] if WANDB_ENABLED else [],  # report to wandb only if enabled
    remove_unused_columns=config["training_args"]["remove_unused_columns"],
    logging_steps=config["training_args"]["logging_steps"],
    max_steps=config["training_args"]["max_steps"],
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
print("🏋️ Initializing GRPO trainer...")
trainer = GRPOTrainer(
    model=model1,
    reward_funcs=[execution_reward_function],
    args=training_args,
    train_dataset=dataset,
)

print("Starting GRPO training for code execution...")

# Initialize wandb run (only if W&B is enabled by user)
if WANDB_ENABLED:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    project_name = f"{config['wandb']['project_name_prefix']}-{timestamp}"
    wandb.init(project=project_name)
    print(
        f"✅ Initialized W&B run: {wandb.run.name} (Offline mode: {offline_mode})"
    )  # Adjusted print message

# Run initial evaluation if enabled
if config["evaluation"]["enabled_initial"] and mbpp_evaluator.should_evaluate(
    is_start=True
):
    print("🧪 Running initial MBPP evaluation...")
    initial_results = mbpp_evaluator.evaluate_model(
        model1, tokenizer1, step=0, phase="initial"
    )

    if (
        WANDB_ENABLED and wandb.run and initial_results.get("enabled", False)
    ):  # Only log if W&B is enabled
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
if config["evaluation"]["enabled_final"] and mbpp_evaluator.should_evaluate(
    is_end=True
):
    print("🧪 Running final MBPP evaluation...")
    final_results = mbpp_evaluator.evaluate_model(
        model1, tokenizer1, step=trainer.state.global_step, phase="final"
    )

    if (
        WANDB_ENABLED and wandb.run and final_results.get("enabled", False)
    ):  # Only log if W&B is enabled
        wandb.log(
            {
                "mbpp_eval/final_pass_rate": final_results["pass_rate"],
                "mbpp_eval/final_problems_passed": final_results["problems_passed"],
                "mbpp_eval/final_total_problems": final_results["total_problems"],
                "mbpp_eval/final_eval_time": final_results["eval_time_seconds"],
            }
        )

print("Code execution training completed!")

# Cleanup vLLM integration
if vllm_integration:
    print("🧹 Cleaning up vLLM server...")
    cleanup_vllm_integration()

if WANDB_ENABLED:  # Only finish if W&B was enabled
    wandb.finish()
    print("✅ Training completed (W&B run finished)")  # Adjusted print message
else:
    print("✅ Training completed (W&B logging was disabled)")  # New message
