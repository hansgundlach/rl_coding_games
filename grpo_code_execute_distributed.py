#!/usr/bin/env python3
"""
Distributed GRPO Code Execution Training - Model learns to generate executable code
Uses PyTorch DDP for distributed training across multiple GPUs
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import re
import io
import contextlib
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
import concurrent.futures
import multiprocessing
import functools


# Initialize distributed training first
def init_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))

        print(f"[Rank {rank}] Initializing distributed training...")
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
    print("🚀 Starting Distributed GRPO Code Execution Training...")
    print("📋 Initializing components...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Distributed GRPO Code Execution Training")
parser.add_argument(
    "--config",
    type=str,
    default="configs/grpo_code_execution.yaml",
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
        for idx, k in enumerate(keys[:-1]):
            if k not in current:
                if dist_info["is_main_process"]:
                    print(f"❌ ERROR: Invalid config override path '{key}'")
                    print(f"   Key '{k}' does not exist in config at level {idx+1}")
                    print(f"   Available keys at this level: {list(current.keys())}")
                sys.exit(1)
            current = current[k]

        # Check if final key exists
        final_key = keys[-1]
        if final_key not in current:
            if dist_info["is_main_process"]:
                print(f"⚠️ WARNING: Creating new config key '{key}'")
                print(f"   Available keys at this level: {list(current.keys())}")

        old_value = current.get(final_key, "NOT_SET")
        current[final_key] = value
        overrides_applied.append(f"{key}: {old_value} -> {value}")

        i += 1

    if overrides_applied and dist_info["is_main_process"]:
        print("🔧 Applied config overrides:")
        for override in overrides_applied:
            print(f"   {override}")

    return config


# Apply any config overrides
config = apply_config_overrides(config, unknown_args)

# Extract config values for easy access - only main process handles W&B
WANDB_ENABLED = config["wandb"]["enabled"] and dist_info["is_main_process"]


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
    print(
        f"🌐 Distributed training: {dist_info['world_size']} GPUs (Rank {dist_info['rank']}/{dist_info['world_size']-1})"
    )

# Extract values for easier use
offline_mode = platform_info["offline_mode"]

# Set global environment variables for transformers library and W&B
if WANDB_ENABLED:  # Check if W&B is enabled by user and is main process
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
        print("🚫 W&B logging disabled (not main process or disabled by user).")

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if dist_info["is_main_process"]:
    print("📦 Loading utility modules...")

from utils.env_loader import get_api_key
from utils.seed_manager import SeedManager
from utils.vllm_client import (
    initialize_vllm_integration,
    cleanup_vllm_integration,
    get_vllm_integration,
)
from evaluation.mbpp.evaluator import MBPPEvaluator, EvalConfig

# Initialize wandb with API key from environment (only on main process)
if WANDB_ENABLED:  # Only try to log in if W&B is enabled and main process
    import wandb

    wandb_key = get_api_key("wandb", required=False)
    if wandb_key:
        wandb.login(key=wandb_key)
        if dist_info["is_main_process"]:
            print("✓ Logged into W&B using environment variable")
    else:
        if dist_info["is_main_process"]:
            print(
                "⚠️ No W&B API key found, continuing with W&B logging (if online) or local saves (if offline)."
            )
else:
    if dist_info["is_main_process"]:
        print("🚫 Skipping W&B login (W&B is disabled or not main process).")

# Initialize comprehensive seed management
if dist_info["is_main_process"]:
    print("🎲 Setting up seed management...")
seed_manager = SeedManager.from_config(config)
seed_manager.seed_everything()

# Initialize MBPP evaluator with consolidated config (only on main process)
mbpp_evaluator = None
if dist_info["is_main_process"]:
    print("🧪 Setting up MBPP evaluator...")

    # Create evaluation config from main config
    eval_config_dict = config.get("evaluation", {}).copy()

    # Remove keys not expected by EvalConfig constructor
    eval_config_dict.pop("enabled_initial", None)
    eval_config_dict.pop("enabled_final", None)
    eval_config_dict.pop("eval_interval_steps", None)
    eval_config_dict.pop("consistent_questions", None)

    # Update results directory to logs folder if running in SLURM
    log_dir = os.environ.get("GRPO_LOG_DIR", "logs")
    if "GRPO_LOG_DIR" in os.environ:
        eval_config_dict["results_dir"] = log_dir
        print(f"📊 Evaluation results will be saved to: {log_dir}")

    # Create EvalConfig object from consolidated config
    eval_config = EvalConfig(**eval_config_dict)

    # Print evaluation config summary
    print("\n" + "=" * 50)
    print("📊 MBPP Evaluation Configuration")
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


def extract_and_execute_code(completion_with_index, timeout=3):
    """Helper function for parallel code execution."""
    i, comp = completion_with_index

    if not isinstance(comp, str):
        return i, -1.0, {"error": "Invalid completion", "success": False}

    # Extract Python code from completion
    code_match = re.search(r"```python\s*\n(.*?)```", comp, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # Try to extract code without markdown formatting
        code = comp.strip()

    if not code:
        return i, -1.0, {"error": "No code generated", "success": False}

    # Execute the code safely
    execution_result = safe_execute_code(code, timeout=timeout)

    return i, comp, code, execution_result


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

if dist_info["is_main_process"]:
    print(f"Created dataset: {dataset}")

# Set up model cache directory
cache_dir = config["model"]["cache_dir"]
os.makedirs(cache_dir, exist_ok=True)

# Use exactly the model specified in config
model_id = config["model"]["id"]
if dist_info["is_main_process"]:
    print(f"📥 Using model from config: {model_id}")
    print(f"📥 Loading trainable model: {model_id}")
    print("⏳ This may take 2-3 minutes depending on model size and storage speed...")

# Load model on each GPU
# Try to use Flash-Attention 2 if available, otherwise fall back to default attention
extra_model_kwargs = {}
try:
    import flash_attn_2_cuda  # noqa: F401

    extra_model_kwargs["attn_implementation"] = "flash_attention_2"
    if dist_info["is_main_process"]:
        print("✅ Flash-Attention 2 detected – using it for faster inference")
except ImportError:
    if dist_info["is_main_process"]:
        print("⚠️ Flash-Attention 2 not installed – falling back to standard attention")

model1 = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map={"": dist_info["local_rank"]},  # Map to specific GPU
    cache_dir=cache_dir,
    local_files_only=offline_mode,
    **extra_model_kwargs,
)

if dist_info["is_main_process"]:
    print("🔤 Loading tokenizer...")
tokenizer1 = AutoTokenizer.from_pretrained(
    model_id, cache_dir=cache_dir, local_files_only=offline_mode
)

# Ensure tokenizer has pad token
if tokenizer1.pad_token is None:
    tokenizer1.pad_token = tokenizer1.eos_token

# Load LoRA
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
model1 = get_peft_model(model1, lora_config)

# Print trainable parameters before wrapping with DDP
if dist_info["is_main_process"]:
    model1.print_trainable_parameters()

# Initialize vLLM integration if enabled (only on main process for safety)
vllm_integration = None
if config.get("vllm", {}).get("enabled", False) and dist_info["is_main_process"]:
    print("🚀 Initializing vLLM integration...")

    # Resolve model path for vLLM, especially in offline mode
    vllm_model_path = model_id
    if offline_mode:
        print("🔧 Offline mode: Resolving local model path for vLLM...")
        cached_model_dirs = glob.glob(
            os.path.join(
                cache_dir, f"models--{model_id.replace('/', '--')}", "snapshots", "*"
            )
        )
        if cached_model_dirs:
            vllm_model_path = cached_model_dirs[0]
            print(f"  ✓ Found local vLLM model path: {vllm_model_path}")
        else:
            print(f"  ⚠️ Could not find local snapshot for {model_id}. vLLM might fail.")

    try:
        vllm_integration = initialize_vllm_integration(
            config["vllm"], vllm_model_path, offline_mode
        )
        if vllm_integration.initialize():
            print("✅ vLLM server started successfully")
        else:
            print("⚠️ vLLM server failed to start, will use HuggingFace fallback")
    except Exception as e:
        print(f"⚠️ vLLM initialization failed: {e}")
        print("   Continuing with HuggingFace generation...")
else:
    if dist_info["is_main_process"]:
        print(
            "📝 vLLM integration disabled or not main process, using HuggingFace generation"
        )


def execution_reward_function(completions, **kwargs):
    """
    Reward function for code execution training.
    Uses CPU parallelism for faster code execution.
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
        and dist_info["is_main_process"]  # Only use vLLM on main process
    ):
        vllm_used = True
        if (
            config.get("vllm", {})
            .get("integration", {})
            .get("log_performance_comparison", False)
        ):
            print(
                f"🚀 [Rank {dist_info['rank']}] Using vLLM for {len(completions)} completions"
            )

    # Determine number of workers - use available CPU cores but not more than completions
    max_workers = min(len(completions), multiprocessing.cpu_count())

    if dist_info["is_main_process"]:
        print(
            f"🔥 [Rank {dist_info['rank']}] Executing {len(completions)} completions in parallel using {max_workers} CPU cores"
        )

    # Prepare completions with indices for parallel processing
    indexed_completions = list(enumerate(completions))

    # Execute all completions in parallel
    execution_results = {}
    start_time = time.time()

    try:
        # Create partial function with timeout from config
        execute_with_timeout = functools.partial(
            extract_and_execute_code, timeout=config["execution"]["timeout"]
        )

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(execute_with_timeout, comp_with_idx): comp_with_idx[0]
                for comp_with_idx in indexed_completions
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    result = future.result()
                    if len(result) == 3:  # Error case
                        i, reward, error_info = result
                        execution_results[i] = (completions[i], "", error_info, reward)
                    else:  # Success case
                        i, comp, code, execution_result = result
                        execution_results[i] = (comp, code, execution_result, None)
                except Exception as e:
                    index = future_to_index[future]
                    execution_results[index] = (
                        completions[index],
                        "",
                        {"error": f"Execution failed: {str(e)}", "success": False},
                        -1.0,
                    )
    except Exception as e:
        # Fallback to sequential execution if parallel fails
        if dist_info["is_main_process"]:
            print(f"⚠️ Parallel execution failed: {e}, falling back to sequential")
        for i, comp in enumerate(completions):
            result = extract_and_execute_code(
                (i, comp), timeout=config["execution"]["timeout"]
            )
            if len(result) == 3:
                execution_results[i] = (comp, "", result[2], result[1])
            else:
                execution_results[i] = (comp, result[2], result[3], None)

    parallel_time = time.time() - start_time

    # Process results in order and calculate rewards
    for i in range(len(completions)):
        comp, code, execution_result, pre_calculated_reward = execution_results[i]

        if pre_calculated_reward is not None:
            # Error case with pre-calculated reward
            reward = pre_calculated_reward
            failed_executions += 1
        else:
            # Calculate reward based on execution result
            if execution_result["success"]:
                if execution_result["output"]:
                    reward = config["rewards"]["success_with_output"]
                    successful_executions += 1
                else:
                    reward = config["rewards"]["success_no_output"]
                    successful_executions += 1
            elif execution_result["timeout"]:
                reward = config["rewards"]["timeout_error"]
                timeout_executions += 1
            elif (
                "SyntaxError" in execution_result["error"]
                or "IndentationError" in execution_result["error"]
            ):
                reward = config["rewards"]["syntax_error"]
                syntax_errors += 1
            else:
                reward = config["rewards"]["runtime_error"]
                failed_executions += 1

        # Debug output (show first few, only on main process)
        if (
            i < config["execution"]["debug_completions"]
            and dist_info["is_main_process"]
        ):
            print("=" * 50)
            print(f"[Rank {dist_info['rank']}] Completion {i+1}:")
            print(f"Code: {code[:200]}...")
            print(f"Success: {execution_result.get('success', False)}")
            print(f"Output: {execution_result.get('output', '')[:100]}")
            print(f"Error: {execution_result.get('error', '')[:100]}")
            print(f"Execution time: {execution_result.get('execution_time', 0):.2f}s")
            print(f"Reward: {reward}")
            print("=" * 50)

        # Show progress for all completions (only on main process)
        if dist_info["is_main_process"]:
            status = "✅" if execution_result.get("success", False) else "❌"
            print(
                f"[Rank {dist_info['rank']}] Completion {i+1}/{len(completions)}: {status} reward={reward:.1f}, time={execution_result.get('execution_time', 0):.2f}s"
            )

        rewards.append(reward)

    if dist_info["is_main_process"]:
        print(
            f"🚀 [Rank {dist_info['rank']}] Parallel execution completed in {parallel_time:.2f}s using {max_workers} workers"
        )

    # Log metrics to wandb after processing the entire batch (only main process)
    if WANDB_ENABLED and dist_info["is_main_process"]:
        import wandb

        if wandb.run:
            avg_batch_reward = sum(rewards) / len(rewards) if rewards else 0
            success_rate = (
                successful_executions / len(completions) if completions else 0
            )

            wandb_metrics = {
                "execution/avg_batch_reward": avg_batch_reward,
                "execution/success_rate": success_rate,
                "execution/successful_executions": successful_executions,
                "execution/failed_executions": failed_executions,
                "execution/timeout_executions": timeout_executions,
                "execution/syntax_errors": syntax_errors,
                "execution/total_completions": len(completions),
                "execution/parallel_time": parallel_time,
                "execution/cpu_workers": max_workers,
                "distributed/rank": dist_info["rank"],
                "distributed/world_size": dist_info["world_size"],
            }

            # Add vLLM performance tracking
            if vllm_used:
                wandb_metrics["vllm/used_for_completions"] = True
                wandb_metrics["vllm/batch_size"] = len(completions)

            wandb.log(wandb_metrics)

    return rewards


# Training arguments - adjust for distributed training and GPU memory
if dist_info["is_main_process"]:
    print("Setting up GRPO training for distributed code execution...")

# Adaptive settings based on GPU type and model size - optimized for distributed training
if platform_info["gpu_type"] == "V100":
    if "3B" in model_id:
        # V100 with 3B model - distributed training allows larger effective batch size
        batch_size = (
            2  # Per GPU batch size - effective batch size will be 4 across 2 GPUs
        )
        gradient_accumulation_steps = 2  # Reduced since we have 2 GPUs
        max_prompt_length = 128
        max_completion_length = 256
        if dist_info["is_main_process"]:
            print(
                "🔧 Using V100 + 3B model settings (distributed, aggressive memory optimization)"
            )
    else:
        # V100 with 1.5B model - can use larger batch sizes with distributed training
        batch_size = (
            4  # Per GPU batch size - effective batch size will be 8 across 2 GPUs
        )
        gradient_accumulation_steps = 1  # No need for gradient accumulation with 2 GPUs
        max_prompt_length = 256
        max_completion_length = 384
        if dist_info["is_main_process"]:
            print(
                "🔧 Using V100 + 1.5B model settings (distributed, moderate memory optimization)"
            )
else:
    # A100 or other GPUs - standard settings but optimized for distributed
    batch_size = 6  # Per GPU batch size - effective batch size will be 12 across 2 GPUs
    gradient_accumulation_steps = 1
    max_prompt_length = 512
    max_completion_length = 512
    if dist_info["is_main_process"]:
        print("🔧 Using standard memory settings for A100/other GPUs (distributed)")

# Calculate effective batch size for logging
effective_batch_size = (
    batch_size * dist_info["world_size"] * gradient_accumulation_steps
)
if dist_info["is_main_process"]:
    print(
        f"💡 Effective batch size: {effective_batch_size} (per_device: {batch_size} × world_size: {dist_info['world_size']} × grad_accum: {gradient_accumulation_steps})"
    )

# Update output directory to logs folder if running in SLURM
output_dir = config["training_args"]["output_dir"]
if "GRPO_LOG_DIR" in os.environ:
    output_dir = os.path.join(os.environ["GRPO_LOG_DIR"], "checkpoints")
    if dist_info["is_main_process"]:
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
    report_to=(
        ["wandb"] if WANDB_ENABLED else []
    ),  # report to wandb only if enabled and main process
    remove_unused_columns=config["training_args"]["remove_unused_columns"],
    logging_steps=config["training_args"]["logging_steps"],
    max_steps=config["training_args"]["max_steps"],
    # Distributed training specific settings
    ddp_find_unused_parameters=False,  # More efficient for our use case
    dataloader_num_workers=2,  # Reduced for V100 memory optimization
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
        if dist_info["is_main_process"]:
            print(f"🔧 Set model config path to: {cached_model_dirs[0]}")

# Create trainer
if dist_info["is_main_process"]:
    print("🏋️ Initializing distributed GRPO trainer...")
trainer = GRPOTrainer(
    model=model1,
    reward_funcs=[execution_reward_function],
    args=training_args,
    train_dataset=dataset,
)

if dist_info["is_main_process"]:
    print("Starting distributed GRPO training for code execution...")

# Initialize wandb run (only if W&B is enabled by user and main process)
if WANDB_ENABLED:
    import wandb

    # Create human-readable timestamp: Jul31_2025_14h30m
    timestamp = datetime.datetime.now().strftime("%b%d_%Y_%Hh%Mm")
    project_name = f"{config['wandb']['project_name_prefix']}-distributed-{timestamp}"
    wandb.init(
        project=project_name,
        config={
            **config,
            **seed_manager.get_seed_info(),
            "distributed": {
                "world_size": dist_info["world_size"],
                "rank": dist_info["rank"],
                "effective_batch_size": effective_batch_size,
            },
        },
    )
    print(
        f"✅ Initialized W&B run: {wandb.run.name} (Project: {project_name}, Offline mode: {offline_mode})"
    )

# Run initial evaluation if enabled (only on main process)
if (
    config["evaluation"].get("enabled_initial", True)
    and mbpp_evaluator is not None
    and mbpp_evaluator.should_evaluate(is_start=True)
    and dist_info["is_main_process"]
):
    print("🧪 Running initial MBPP evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("initial")
    initial_results = mbpp_evaluator.evaluate_model(
        model1, tokenizer1, step=0, phase="initial"
    )

    # Log to W&B if enabled
    if WANDB_ENABLED and "pass_rate" in initial_results:
        import wandb

        if wandb.run:
            wandb.log(
                {
                    "mbpp_eval/initial_pass_rate": initial_results["pass_rate"],
                    "mbpp_eval/initial_problems_passed": initial_results[
                        "problems_passed"
                    ],
                    "mbpp_eval/initial_total_problems": initial_results[
                        "total_problems"
                    ],
                    "mbpp_eval/initial_eval_time": initial_results["eval_time_seconds"],
                }
            )


# Add interval evaluation callback (only on main process)
class DistributedIntervalEvaluationCallback(TrainerCallback):
    def __init__(
        self,
        evaluator,
        model,
        tokenizer,
        config,
        wandb_enabled,
        seed_manager,
        dist_info,
    ):
        self.evaluator = evaluator
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.wandb_enabled = wandb_enabled
        self.seed_manager = seed_manager
        self.dist_info = dist_info
        self.eval_interval = config["evaluation"].get("eval_interval_steps", None)

    def on_step_end(self, args, state, control, **kwargs):
        # Only run evaluation on main process
        if not self.dist_info["is_main_process"]:
            return

        # Run interval evaluation if enabled and it's time
        if (
            self.eval_interval
            and self.evaluator is not None
            and self.evaluator.config.enabled
            and state.global_step > 0  # Skip step 0 (initial eval already done)
            and state.global_step % self.eval_interval == 0
        ):
            print(
                f"🧪 [Main Process] Running interval MBPP evaluation at step {state.global_step}..."
            )
            # Seed for consistent evaluation
            self.seed_manager.seed_for_evaluation_auto(
                f"interval_step_{state.global_step}"
            )
            interval_results = self.evaluator.evaluate_model(
                self.model, self.tokenizer, step=state.global_step, phase="interval"
            )

            if self.wandb_enabled and "pass_rate" in interval_results:
                import wandb

                if wandb.run:
                    wandb.log(
                        {
                            "mbpp_eval/pass_rate": interval_results["pass_rate"],
                            "mbpp_eval/problems_passed": interval_results[
                                "problems_passed"
                            ],
                            "mbpp_eval/total_problems": interval_results[
                                "total_problems"
                            ],
                            "mbpp_eval/eval_time": interval_results[
                                "eval_time_seconds"
                            ],
                            "step": state.global_step,
                        }
                    )


# Add the callback to trainer (pass evaluator only to main process)
interval_callback = DistributedIntervalEvaluationCallback(
    mbpp_evaluator if dist_info["is_main_process"] else None,
    model1,
    tokenizer1,
    config,
    WANDB_ENABLED,
    seed_manager,
    dist_info,
)
trainer.add_callback(interval_callback)

# Train model
if dist_info["is_main_process"]:
    print("🏃 Starting distributed training...")
trainer.train()

# Run final evaluation if enabled (only on main process)
if (
    config["evaluation"].get("enabled_final", True)
    and mbpp_evaluator is not None
    and mbpp_evaluator.should_evaluate(is_end=True)
    and dist_info["is_main_process"]
):
    print("🧪 Running final MBPP evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("final")
    final_results = mbpp_evaluator.evaluate_model(
        model1, tokenizer1, step=trainer.state.global_step, phase="final"
    )

    if WANDB_ENABLED and "pass_rate" in final_results:
        import wandb

        if wandb.run:
            wandb.log(
                {
                    "mbpp_eval/final_pass_rate": final_results["pass_rate"],
                    "mbpp_eval/final_problems_passed": final_results["problems_passed"],
                    "mbpp_eval/final_total_problems": final_results["total_problems"],
                    "mbpp_eval/final_eval_time": final_results["eval_time_seconds"],
                }
            )

if dist_info["is_main_process"]:
    print("Code execution training completed!")

# Cleanup vLLM integration (only on main process)
if vllm_integration and dist_info["is_main_process"]:
    print("🧹 Cleaning up vLLM server...")
    cleanup_vllm_integration()

# Clean up distributed training
if dist_info["world_size"] > 1:
    dist.destroy_process_group()

if (
    WANDB_ENABLED and dist_info["is_main_process"]
):  # Only finish if W&B was enabled and main process
    import wandb

    wandb.finish()
    print("✅ Distributed training completed (W&B run finished)")
else:
    if dist_info["is_main_process"]:
        print(
            "✅ Distributed training completed (W&B logging was disabled or not main process)"
        )
