#!/usr/bin/env python3
"""
GRPO Code Generation Game with ICL Memory Opponent

Training regime:
- Player 1 (Generator): Trained with GRPO, weights update
- Player 2 (Guesser): Frozen weights + ICL memory that accumulates winning examples
- Memory snapshots for training stability (80% latest ICL, 20% frozen snapshots)

Game Rules:
- Player 1 generates code and predicts its output
- Player 2 guesses the output using ICL memory of past winning examples
- Player 1 wins if code executable AND self-prediction correct AND Player 2 wrong
- Player 2 wins if prediction correct
"""

# Set environment variable to avoid tokenizers warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
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
from collections import deque
from copy import deepcopy

print("🚀 Starting GRPO Code Game with ICL Memory Opponent...")
print("📋 Initializing components...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="GRPO Code Game with ICL Memory")
parser.add_argument(
    "--config",
    type=str,
    default="configs/grpo_code_game_icl.yaml",
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

mbpp_evaluator = MBPPEvaluator(eval_config)

if not mbpp_evaluator.config.enabled:
    print("⚠️ MBPP evaluation disabled - dataset not found")
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
    import subprocess
    import tempfile
    import time

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

@dataclass
class WinningExample:
    """A winning example for ICL memory."""
    code: str
    expected_output: str
    brief_rationale: str

class ICLMemory:
    """In-Context Learning memory that stores winning examples."""
    
    def __init__(self, max_size: int = 16):
        self.max_size = max_size
        self.examples: List[WinningExample] = []
    
    def add_winning_example(self, code: str, expected_output: str, brief_rationale: str):
        """Add a winning example to memory."""
        example = WinningExample(code, expected_output, brief_rationale)
        self.examples.append(example)
        
        # Keep only the most recent examples
        if len(self.examples) > self.max_size:
            self.examples = self.examples[-self.max_size:]
    
    def add_and_prune(self, wins: List[Dict], k: int = 16):
        """Add multiple wins and prune to k examples."""
        for win in wins:
            self.add_winning_example(
                win["code"], 
                win["expected_output"], 
                win.get("brief_rationale", "Code executed successfully")
            )
        
        # Prune to k most recent
        if len(self.examples) > k:
            self.examples = self.examples[-k:]
    
    def get_icl_prompt_prefix(self) -> str:
        """Generate ICL prompt prefix with examples."""
        if not self.examples:
            return ""
        
        prefix = "Here are some examples of code and their outputs:\n\n"
        for i, example in enumerate(self.examples[-8:]):  # Use last 8 examples
            prefix += f"Example {i+1}:\n"
            prefix += f"Code:\n```python\n{example.code}\n```\n"
            prefix += f"Output: {example.expected_output}\n"
            prefix += f"Rationale: {example.brief_rationale}\n\n"
        
        return prefix + "Now predict the output for the following code:\n\n"

class ICLOpponent:
    """ICL-based opponent that uses frozen weights + memory."""
    
    def __init__(self, model, tokenizer, device, memory: ICLMemory):
        self.model = model
        self.tokenizer = tokenizer  
        self.device = device
        self.memory = memory
    
    def predict_output(self, code: str) -> str:
        """Predict output using ICL memory."""
        # Build prompt with ICL examples
        icl_prefix = self.memory.get_icl_prompt_prefix()
        
        prompt = f"""{icl_prefix}Code:
```python
{code}
```

Think step by step about what this code does:
1. Analyze each line of code
2. Trace through the execution
3. Determine the final output

Provide your prediction in this exact format:
<prediction>
[exact output here]
</prediction>

What will this code output?"""
        
        # Generate prediction
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config["generation"]["guesser_max_tokens"],
                temperature=config["generation"]["temperature"],
                top_p=config["generation"]["top_p"],
                top_k=(
                    config["generation"]["top_k"]
                    if config["generation"]["top_k"] > 0
                    else None
                ),
                do_sample=config["generation"]["do_sample"],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        # Extract prediction
        pred_match = re.search(r"<prediction>\s*(.*?)\s*</prediction>", response, re.DOTALL)
        if pred_match:
            return pred_match.group(1).strip()
        
        return ""

# Memory management for training stability
REFRESH_EVERY = config["icl"]["refresh_every"]  # games between ICL memory updates
SNAPSHOT_MAX = config["icl"]["snapshot_max"]    # how many frozen memories to keep
P_LATEST = config["icl"]["p_latest"]            # 80% latest ICL, 20% snapshot

snapshot_buf = deque(maxlen=SNAPSHOT_MAX)

def save_snapshot(latest_memory: ICLMemory):
    """Save a frozen copy of the memory."""
    snapshot_buf.append(deepcopy(latest_memory))

def sample_opponent(generator_model, guesser_model, guesser_tokenizer, device, latest_memory: ICLMemory) -> ICLOpponent:
    """Sample opponent: 80% latest ICL, 20% frozen snapshot."""
    if not snapshot_buf or random.random() < P_LATEST:
        return ICLOpponent(guesser_model, guesser_tokenizer, device, latest_memory)  # latest ICL
    return ICLOpponent(guesser_model, guesser_tokenizer, device, random.choice(snapshot_buf))  # frozen snapshot

@dataclass
class CodeGameTrajectory:
    """Store trajectory data for a single code generation game."""
    
    generator_data: Dict  # Player 1 (generates code + predicts output)
    game_outcome: Dict    # Final results and rewards
    execution_result: Dict # Code execution results

def extract_code_from_response(response: str) -> str:
    """Extract Python code from model response."""
    # Look for code in markdown format
    code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # Fallback: return the response as-is
    return response.strip()

def extract_prediction_from_response(response: str) -> str:
    """Extract prediction from model response."""
    # Look for prediction tags
    pred_match = re.search(r"<prediction>\s*(.*?)\s*</prediction>", response, re.DOTALL)
    if pred_match:
        return pred_match.group(1).strip()
    
    # Fallback: return empty string if no prediction found
    return ""

def play_code_game(generator_model, guesser_opponent: ICLOpponent, tokenizer, device) -> CodeGameTrajectory:
    """
    Play a single game between generator (GRPO-trained) and guesser (ICL).
    
    Returns complete trajectory data for training.
    """
    # Player 1: Generate code AND predict output
    generator_prompt = """You are Player 1 in a code generation game. Your goals are to:
1. Write Python code that executes successfully without errors
2. Correctly predict what your code will output
3. Write code whose output Player 2 will struggle to predict correctly

Write a complete Python program that demonstrates a programming concept or solves a simple problem.
The program should be executable and produce some output.

Focus on writing clean, working code that has interesting or non-obvious output that you can predict but Player 2 cannot.

Format your response with the Python code AND your prediction:
```python
# Your code here
```

<prediction>
[exact output your code will produce]
</prediction>

Write a Python program and predict its output:"""

    inputs = tokenizer(
        generator_prompt, return_tensors="pt", truncation=True, max_length=512
    )
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generator_outputs = generator_model.generate(
            **inputs,
            max_new_tokens=config["generation"]["generator_max_tokens"],
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

    generator_response = tokenizer.decode(
        generator_outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    # Extract code and prediction from generator response
    generator_code = extract_code_from_response(generator_response)
    generator_prediction = extract_prediction_from_response(generator_response)

    # Execute the generated code to get actual output
    execution_result = safe_execute_code(generator_code, config["game"]["timeout"])
    actual_output = execution_result["output"] if execution_result["success"] else ""

    # Player 2: Guess the output using ICL
    guesser_prediction = guesser_opponent.predict_output(generator_code)

    # Calculate rewards
    rewards = calculate_rewards(execution_result, generator_prediction, guesser_prediction, actual_output)

    return CodeGameTrajectory(
        generator_data={
            "prompt": generator_prompt,
            "response": generator_response,
            "code": generator_code,
            "prediction": generator_prediction,
            "role": "generator",
        },
        game_outcome={
            "generator_reward": rewards["generator"],
            "guesser_reward": rewards["guesser"],
            "code_executable": execution_result["success"],
            "generator_prediction_correct": rewards["generator_prediction_correct"],
            "guesser_prediction_correct": rewards["guesser_prediction_correct"],
            "actual_output": actual_output,
        },
        execution_result=execution_result,
    )

def calculate_rewards(execution_result: Dict, generator_prediction: str, guesser_prediction: str, actual_output: str) -> Dict:
    """
    Calculate rewards for both players.
    
    Generator: Gets positive reward if:
    - Code is executable AND
    - Their prediction is correct AND
    - Player 2's prediction is wrong
    
    Guesser: Gets positive reward if:
    - Their prediction is correct
    """
    # Check if code is executable
    code_executable = execution_result["success"]
    
    # Check prediction correctness (exact match)
    generator_prediction_correct = (
        generator_prediction.strip() == actual_output.strip()
        if code_executable
        else False
    )
    
    guesser_prediction_correct = (
        guesser_prediction.strip() == actual_output.strip()
        if code_executable
        else False
    )
    
    # Generator: +1 if (executable AND self-correct AND guesser-wrong), -1 otherwise
    generator_reward = (
        1.0
        if (
            code_executable
            and generator_prediction_correct
            and not guesser_prediction_correct
        )
        else -1.0
    )
    
    # Guesser: +1 if correct prediction, -1 otherwise
    guesser_reward = 1.0 if guesser_prediction_correct else -1.0
    
    return {
        "generator": generator_reward,
        "guesser": guesser_reward,
        "generator_prediction_correct": generator_prediction_correct,
        "guesser_prediction_correct": guesser_prediction_correct,
    }

# GRPO reward function
def grpo_reward_function(completions, **kwargs):
    """
    Reward function for GRPO training based on game outcomes.
    This integrates with the ICL memory game results.
    """
    # Get the latest game results from the global storage
    if hasattr(grpo_reward_function, 'latest_rewards'):
        rewards = grpo_reward_function.latest_rewards[:len(completions)]
        # Pad with zeros if needed
        while len(rewards) < len(completions):
            rewards.append(0.0)
        return rewards
    else:
        # Default rewards if no game results available
        return [0.0] * len(completions)

# Set up model cache directory
cache_dir = config["generator_model"]["cache_dir"]
os.makedirs(cache_dir, exist_ok=True)

# Load generator model (trainable with GRPO)
generator_model_id = config["generator_model"]["id"]
print(f"📥 Loading generator model: {generator_model_id}")

generator_model = AutoModelForCausalLM.from_pretrained(
    generator_model_id,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_dir,
    local_files_only=offline_mode,
)
generator_tokenizer = AutoTokenizer.from_pretrained(
    generator_model_id, cache_dir=cache_dir, local_files_only=offline_mode
)

# Ensure tokenizer has pad token
if generator_tokenizer.pad_token is None:
    generator_tokenizer.pad_token = generator_tokenizer.eos_token

# Load guesser model (frozen weights for ICL)
guesser_model_id = config["guesser_model"]["id"]
print(f"📥 Loading guesser model (frozen): {guesser_model_id}")

guesser_model = AutoModelForCausalLM.from_pretrained(
    guesser_model_id,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_dir,
    local_files_only=offline_mode,
)
guesser_tokenizer = AutoTokenizer.from_pretrained(
    guesser_model_id, cache_dir=cache_dir, local_files_only=offline_mode
)

# Ensure guesser tokenizer has pad token
if guesser_tokenizer.pad_token is None:
    guesser_tokenizer.pad_token = guesser_tokenizer.eos_token

# Freeze guesser model weights
for param in guesser_model.parameters():
    param.requires_grad = False

print("🔒 Guesser model weights frozen for ICL")

# Add LoRA to generator model only
print("🔧 Setting up LoRA configuration for generator...")
lora_config = LoraConfig(
    task_type=config["lora"]["task_type"],
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["lora_alpha"],
    target_modules=config["lora"]["target_modules"],
)
print("🎯 Applying LoRA to generator model...")
generator_model = get_peft_model(generator_model, lora_config)
print(generator_model.print_trainable_parameters())

# Get device
device = next(generator_model.parameters()).device
print(f"Model device: {device}")

# Initialize ICL memory
latest_memory = ICLMemory(max_size=config["icl"]["memory_size"])
print(f"🧠 Initialized ICL memory with size {config['icl']['memory_size']}")

# Create a simple dataset for GRPO (we'll replace with actual data during training)
prompt = "Generate code and predict output"
dataset_size = config["dataset"]["size"]
dataset = Dataset.from_dict({"prompt": [prompt] * dataset_size})

# Training parameters
num_steps = config["training"]["num_steps"]
games_per_step = config["training"]["games_per_step"]

# Initialize GRPO trainer
training_args = config["training_args"]

# Adaptive settings based on GPU
if platform_info["gpu_type"] == "V100":
    training_args["per_device_train_batch_size"] = min(training_args["per_device_train_batch_size"], 2)
    training_args["gradient_accumulation_steps"] = max(training_args.get("gradient_accumulation_steps", 1), 4)
    print("🔧 Adjusted training args for V100 memory constraints")

grpo_config = GRPOConfig(**training_args)

grpo_trainer = GRPOTrainer(
    model=generator_model,
    tokenizer=generator_tokenizer,
    config=grpo_config,
    reward_funcs=[grpo_reward_function],
    args=grpo_config,
    train_dataset=dataset,
)

print("🎮 Starting GRPO Code Game with ICL Memory Training")

# Initialize wandb run (only if W&B is enabled by user)
if WANDB_ENABLED:
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
        generator_model, generator_tokenizer, step=0, phase="initial"
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

# Custom GRPO training function that integrates ICL games
def run_grpo_training_with_icl():
    """Run GRPO training integrated with ICL memory games."""
    
    # Convert games to prompts for GRPO dataset
    def create_game_dataset():
        game_prompts = []
        for _ in range(games_per_step * 2):  # More prompts than games for variety
            game_prompts.append("""You are Player 1 in a code generation game. Your goals are to:
1. Write Python code that executes successfully without errors
2. Correctly predict what your code will output
3. Write code whose output Player 2 will struggle to predict correctly

Write a complete Python program that demonstrates a programming concept or solves a simple problem.
The program should be executable and produce some output.

Focus on writing clean, working code that has interesting or non-obvious output that you can predict but Player 2 cannot.

Format your response with the Python code AND your prediction:
```python
# Your code here
```

<prediction>
[exact output your code will produce]
</prediction>

Write a Python program and predict its output:""")
        
        return Dataset.from_dict({"prompt": game_prompts})
    
    # Create dataset for GRPO
    training_dataset = create_game_dataset()
    
    # Update trainer dataset
    grpo_trainer.train_dataset = training_dataset
    
    # Training loop with ICL memory management
    games_played = 0
    pending_wins = []  # Collect wins between memory refreshes
    
    for step in range(num_steps):
        print(f"\n🎯 Step {step + 1}/{num_steps}")
        
        # Play games to generate rewards for GRPO
        trajectories = []
        generator_wins = 0
        guesser_wins = 0
        code_executable_count = 0
        rewards_for_grpo = []
        
        for game_idx in range(games_per_step):
            # Sample opponent (80/20 mix)
            opponent = sample_opponent(generator_model, guesser_model, guesser_tokenizer, device, latest_memory)
            
            # Play game
            trajectory = play_code_game(generator_model, opponent, generator_tokenizer, device)
            trajectories.append(trajectory)
            
            # Collect reward for GRPO
            rewards_for_grpo.append(trajectory.game_outcome["generator_reward"])
            
            games_played += 1
            
            # Track statistics
            if trajectory.game_outcome["generator_reward"] > 0:
                generator_wins += 1
                
                # Collect winning example for memory update
                if trajectory.game_outcome["code_executable"] and trajectory.game_outcome["generator_prediction_correct"]:
                    pending_wins.append({
                        "code": trajectory.generator_data["code"],
                        "expected_output": trajectory.game_outcome["actual_output"],
                        "brief_rationale": "Generator won: code executed correctly and prediction was accurate"
                    })
            else:
                guesser_wins += 1
            
            if trajectory.game_outcome["code_executable"]:
                code_executable_count += 1
            
            # Show detailed results for first few games
            if game_idx < 2:  # Show first 2 games
                print(f"\n{'='*60}")
                print(f"🎮 Game {game_idx + 1}/{games_per_step} - Step {step + 1}")
                print(f"{'='*60}")
                
                print("🤖 Generated Code:")
                print("```python")
                print(trajectory.generator_data["code"])
                print("```")
                
                exec_result = trajectory.execution_result
                print(f"\n🔧 Execution Results:")
                print(f"   Success: {'✅' if exec_result['success'] else '❌'}")
                if exec_result["success"]:
                    print(f"   Actual Output: '{trajectory.game_outcome['actual_output']}'")
                else:
                    print(f"   Error: {exec_result['error'][:100]}...")
                
                print(f"\n🎯 Predictions:")
                print(f"   Generator predicted: '{trajectory.generator_data['prediction']}'")
                print(f"   Generator correct: {'✅' if trajectory.game_outcome['generator_prediction_correct'] else '❌'}")
                
                print(f"\n🏆 Rewards:")
                print(f"   Generator: {trajectory.game_outcome['generator_reward']:+.1f}")
                print(f"   Guesser: {trajectory.game_outcome['guesser_reward']:+.1f}")
                print(f"{'='*60}")
        
        # Store rewards for GRPO reward function
        grpo_reward_function.latest_rewards = rewards_for_grpo
        
        # Run GRPO training step
        print(f"🏋️ Running GRPO training step with {len(rewards_for_grpo)} game rewards...")
        grpo_trainer.train()
        
        # Batch ICL memory updates
        if games_played % REFRESH_EVERY == 0 and pending_wins:
            print(f"🧠 Updating ICL memory with {len(pending_wins)} new winning examples...")
            latest_memory.add_and_prune(pending_wins, k=config["icl"]["memory_size"])
            save_snapshot(latest_memory)
            pending_wins = []  # Reset
            print(f"📸 Saved memory snapshot ({len(snapshot_buf)}/{SNAPSHOT_MAX} snapshots)")
        
        # Logging
        executable_rate = code_executable_count / games_per_step
        generator_prediction_accuracy = sum(1 for t in trajectories if t.game_outcome["generator_prediction_correct"]) / games_per_step
        avg_reward = sum(rewards_for_grpo) / len(rewards_for_grpo) if rewards_for_grpo else 0.0
        
        stats = {
            "step": step,
            "generator_wins": generator_wins,
            "guesser_wins": guesser_wins,
            "executable_rate": executable_rate,
            "generator_prediction_accuracy": generator_prediction_accuracy,
            "icl_memory_size": len(latest_memory.examples),
            "snapshot_count": len(snapshot_buf),
            "games_played": games_played,
            "avg_game_reward": avg_reward,
        }
        
        print(f"🏆 Generator wins: {generator_wins}, Guesser wins: {guesser_wins}")
        print(f"💻 Executable code rate: {executable_rate:.2%}")
        print(f"🎯 Generator prediction accuracy: {generator_prediction_accuracy:.2%}")
        print(f"🧠 ICL memory size: {len(latest_memory.examples)}")
        print(f"💰 Average game reward: {avg_reward:.2f}")
        
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
                generator_model, generator_tokenizer, step=step + 1, phase="interval"
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
            generator_model.save_pretrained(checkpoint_dir)
            generator_tokenizer.save_pretrained(checkpoint_dir)
            
            # Save ICL memory state
            with open(f"{checkpoint_dir}/icl_memory.json", "w") as f:
                memory_data = {
                    "examples": [{"code": ex.code, "expected_output": ex.expected_output, "brief_rationale": ex.brief_rationale} 
                               for ex in latest_memory.examples],
                    "games_played": games_played
                }
                json.dump(memory_data, f, indent=2)
            
            print(f"💾 Saved checkpoint at step {step + 1}")

# Run the integrated training
run_grpo_training_with_icl()

print("🏁 GRPO Code Game with ICL Memory training completed!")

# Run final MBPP evaluation if enabled
if config["evaluation"].get("enabled_final", True) and mbpp_evaluator.config.enabled:
    print("🧪 Running final MBPP evaluation...")
    final_results = mbpp_evaluator.evaluate_model(
        generator_model, generator_tokenizer, step=num_steps, phase="final"
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
generator_model.save_pretrained(final_checkpoint_dir)
generator_tokenizer.save_pretrained(final_checkpoint_dir)

# Save final ICL memory state
with open(f"{final_checkpoint_dir}/icl_memory.json", "w") as f:
    memory_data = {
        "examples": [{"code": ex.code, "expected_output": ex.expected_output, "brief_rationale": ex.brief_rationale} 
                   for ex in latest_memory.examples],
        "games_played": games_played
    }
    json.dump(memory_data, f, indent=2)

print(f"💾 Saved final checkpoint")

if WANDB_ENABLED and wandb.run:
    wandb.finish()
    print("✅ Training completed (W&B run finished)")
else:
    print("✅ Training completed (W&B logging was disabled)")