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

        # Set WANDB_RUN_ID before importing wandb to control offline directory name
        timestamp = datetime.datetime.now().strftime("%b%d_%Y_%Hh%Mm")
        run_id = f"grpo-code-game-icl-{timestamp.replace('_', '-').replace('h', 'h-').replace('m', 'm')}"
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

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("📦 Loading utility modules...")
from utils.env_loader import get_api_key
from utils.seed_manager import SeedManager
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

# Initialize comprehensive seed management
print("🎲 Setting up seed management...")
seed_manager = SeedManager.from_config(config)
seed_manager.seed_everything()

# Initialize MBPP evaluator with consolidated config
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

    def add_winning_example(
        self, code: str, expected_output: str, brief_rationale: str
    ):
        """Add a winning example to memory."""
        example = WinningExample(code, expected_output, brief_rationale)
        self.examples.append(example)

        # Keep only the most recent examples
        if len(self.examples) > self.max_size:
            self.examples = self.examples[-self.max_size :]

    def add_and_prune(self, wins: List[Dict], k: int = 16):
        """Add multiple wins and prune to k examples."""
        for win in wins:
            self.add_winning_example(
                win["code"],
                win["expected_output"],
                win.get("brief_rationale", "Code executed successfully"),
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

        prompt = f"""{icl_prefix}<|im_start|>system
You are a concise code analyzer. Predict the output of Python code that produces a list of 2-50 numbers. Be direct and brief - NO THINKING, NO EXPLANATIONS, NO VERBOSE REASONING. Just analyze the code and provide your prediction immediately. Do not use <think> tags or explain your reasoning.
<|im_end|>
<|im_start|>user
Code:
```python
{code}
```

IMPORTANT FORMAT REQUIREMENT:
The code should output a list of numbers that is between 2 and 50 numbers long in the format specified below. 

Examples of valid predictions:
- [1, 2, 3, 4, 5]
- [3.14, 2.71, 1.41]
- [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

Provide your prediction in this exact format. THIS IS VERY IMPORTANT:
<prediction>
[exact list here, e.g., [1, 2, 3, 4, 5]]
</prediction>

What list of numbers will this code output?
<|im_end|>
<|im_start|>assistant
"""

        # Think step by step about what this code does:
        # 1. Analyze each line of code
        # 2. Trace through the execution
        # 3. Determine the final list output
        # 4. Count the numbers in your predicted list (must be 2-50)

        # Generate prediction
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Seed for deterministic guesser response
        seed_manager.seed_for_generation(step=0, generation_idx=hash(code) % 1000)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config["generation"]["guesser_max_tokens"],
                temperature=config["generation"].get(
                    "guesser_temperature", config["generation"]["temperature"]
                ),
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

        # Track step counter for logging
        if not hasattr(self, "step_counter"):
            self.step_counter = 0

        # Increment step counter and log periodically
        self.step_counter += 1

        # Extract prediction
        pred_match = re.search(
            r"<prediction>\s*(.*?)\s*</prediction>", response, re.DOTALL
        )

        # Log full response based on debug configuration
        debug_config = config.get("debug", {})
        show_full_responses = debug_config.get("show_full_responses", False)

        # Log full response every 10 predictions OR if show_full_responses is enabled
        if show_full_responses or self.step_counter % 10 == 0:
            print(f"\n🔍 ICL GUESSER FULL RESPONSE (Prediction #{self.step_counter}):")
            print(f"Code Input: {code[:200]}{'...' if len(code) > 200 else ''}")
            print(f"ICL Guesser Full Response: '{response}'")
            print(f"Response Length: {len(response)} chars")
            print(
                f"Extracted Prediction: '{pred_match.group(1).strip() if pred_match else 'NONE'}'"
            )

        if pred_match:
            return pred_match.group(1).strip()

        return ""


# Memory management for training stability
REFRESH_EVERY = config["icl"]["refresh_every"]  # games between ICL memory updates
SNAPSHOT_MAX = config["icl"]["snapshot_max"]  # how many frozen memories to keep
P_LATEST = config["icl"]["p_latest"]  # 80% latest ICL, 20% snapshot

snapshot_buf = deque(maxlen=SNAPSHOT_MAX)


def save_snapshot(latest_memory: ICLMemory):
    """Save a frozen copy of the memory."""
    snapshot_buf.append(deepcopy(latest_memory))


def sample_opponent(
    generator_model, guesser_model, guesser_tokenizer, device, latest_memory: ICLMemory
) -> ICLOpponent:
    """Sample opponent: 80% latest ICL, 20% frozen snapshot."""
    if not snapshot_buf or random.random() < P_LATEST:
        return ICLOpponent(
            guesser_model, guesser_tokenizer, device, latest_memory
        )  # latest ICL
    return ICLOpponent(
        guesser_model, guesser_tokenizer, device, random.choice(snapshot_buf)
    )  # frozen snapshot


@dataclass
class CodeGameTrajectory:
    """Store trajectory data for a single code generation game."""

    generator_data: Dict  # Player 1 (generates code + predicts output)
    game_outcome: Dict  # Final results and rewards
    execution_result: Dict  # Code execution results


def validate_number_list_format(output_str: str) -> dict:
    """
    Validate that output is a list of numbers between 2 and 50 numbers long.

    Returns:
        dict: {
            'is_valid': bool,
            'error_message': str,
            'parsed_list': List[float] or None,
            'list_length': int
        }
    """
    result = {
        "is_valid": False,
        "error_message": "",
        "parsed_list": None,
        "list_length": 0,
    }

    if not output_str.strip():
        result["error_message"] = "Empty output"
        return result

    try:
        # Try to parse as a Python list/array
        import ast
        import re

        # Clean the output string - remove extra whitespace and newlines
        cleaned = output_str.strip()

        # Try different common list formats
        possible_formats = [
            cleaned,  # As is
            (
                f"[{cleaned}]"
                if not (cleaned.startswith("[") and cleaned.endswith("]"))
                else cleaned
            ),  # Wrap in brackets
            cleaned.replace("(", "[").replace(
                ")", "]"
            ),  # Convert parentheses to brackets
        ]

        parsed_list = None
        for format_attempt in possible_formats:
            try:
                # Use ast.literal_eval for safe evaluation
                parsed_list = ast.literal_eval(format_attempt)
                if isinstance(parsed_list, (list, tuple)):
                    parsed_list = list(parsed_list)
                    break
                elif isinstance(parsed_list, (int, float)):
                    # Single number - convert to list
                    parsed_list = [parsed_list]
                    break
            except:
                continue

        # If ast.literal_eval failed, try regex parsing for space/comma separated numbers
        if parsed_list is None:
            # Extract numbers using regex
            number_pattern = r"-?\d+\.?\d*"
            numbers = re.findall(number_pattern, cleaned)
            if numbers:
                try:
                    parsed_list = [float(num) for num in numbers]
                except:
                    pass

        if parsed_list is None:
            result["error_message"] = "Could not parse as list of numbers"
            return result

        # Validate all elements are numbers
        if not all(isinstance(x, (int, float)) for x in parsed_list):
            result["error_message"] = "Contains non-numeric values"
            return result

        # Check length constraint (2-50 numbers)
        list_length = len(parsed_list)
        if list_length < 2:
            result["error_message"] = (
                f"List too short: {list_length} numbers (minimum 2)"
            )
            result["list_length"] = list_length
            return result
        elif list_length > 50:
            result["error_message"] = (
                f"List too long: {list_length} numbers (maximum 50)"
            )
            result["list_length"] = list_length
            return result

        # All validations passed
        result["is_valid"] = True
        result["parsed_list"] = parsed_list
        result["list_length"] = list_length

    except Exception as e:
        result["error_message"] = f"Parsing error: {str(e)}"

    return result


def normalize_number_list_output(output_str: str) -> str:
    """
    Normalize a number list output to a standard format for comparison.

    Returns the list as a string in format: "[1.0, 2.0, 3.0]"
    """
    validation = validate_number_list_format(output_str)
    if validation["is_valid"]:
        # Convert to consistent format with 1 decimal place
        normalized = [round(float(x), 1) for x in validation["parsed_list"]]
        return str(normalized)
    else:
        return output_str.strip()


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


def play_code_game(
    generator_model, guesser_opponent: ICLOpponent, tokenizer, device
) -> CodeGameTrajectory:
    """
    Play a single game between generator (GRPO-trained) and guesser (ICL).

    Returns complete trajectory data for training.
    """
    # Player 1: Generate code AND predict output
    generator_prompt = """You are Player 1 in a code generation game. Your goals are to:
1. Write Python code that executes successfully without errors
2. Correctly predict what your code will output
3. Write code whose output Player 2 will struggle to predict correctly

IMPORTANT FORMAT REQUIREMENT:
Your code MUST output a list of numbers that is between 2 and 50 numbers long.
Examples of valid outputs:
- [1, 2, 3, 4, 5]
- [3.14, 2.71, 1.41]
- [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

Write a complete Python program that generates and prints a list of numbers.
The program should be executable and produce a list output in the required format.

Focus on writing clean, working code that produces an interesting or non-obvious list that you can predict but Player 2 cannot.



Format your response with the Python code AND your prediction:
```python
# Your code here - must print a list of 2-50 numbers
print([...])  # Your list here
```

<prediction>
[exact list your code will produce, e.g., [1, 2, 3, 4, 5]]
</prediction>

Write a Python program that outputs a list of numbers and predict its output:"""

    # Examples of good strategies:
    # - Mathematical sequences (Fibonacci, primes, factorials)
    # - Calculations with specific numeric results
    # - List comprehensions with non-obvious patterns
    # - String-to-number conversions
    # - Date/time calculations that produce numbers

    inputs = tokenizer(
        generator_prompt, return_tensors="pt", truncation=True, max_length=512
    )
    if device.type == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Seed for deterministic generator response based on prompt content
    seed_manager.seed_for_generation(
        step=0, generation_idx=hash(generator_prompt) % 1000
    )

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
    rewards = calculate_rewards(
        execution_result, generator_prediction, guesser_prediction, actual_output
    )

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
            "output_format_valid": rewards["output_format_valid"],
            "generator_prediction_format_valid": rewards[
                "generator_prediction_format_valid"
            ],
            "guesser_prediction_format_valid": rewards[
                "guesser_prediction_format_valid"
            ],
            "output_validation_error": rewards["output_validation_error"],
            "generator_prediction_validation_error": rewards[
                "generator_prediction_validation_error"
            ],
            "guesser_prediction_validation_error": rewards[
                "guesser_prediction_validation_error"
            ],
        },
        execution_result=execution_result,
    )


def calculate_rewards(
    execution_result: Dict,
    generator_prediction: str,
    guesser_prediction: str,
    actual_output: str,
) -> Dict:
    """
    Calculate rewards for both players with format validation penalties.

    Generator: Gets positive reward if:
    - Code is executable AND
    - Output follows list format (2-50 numbers) AND
    - Their prediction is correct AND follows format AND
    - Player 2's prediction is wrong

    Guesser: Gets positive reward if:
    - Their prediction is correct AND follows format

    FORMAT PENALTIES:
    - Heavy penalty (-1.0) for wrong format
    - Code that doesn't produce valid list format gets -1.0
    - Predictions that don't follow list format get -1.0
    """
    # Check if code is executable
    code_executable = execution_result["success"]

    # Validate output format
    output_validation = (
        validate_number_list_format(actual_output)
        if code_executable
        else {"is_valid": False, "error_message": "Code not executable"}
    )
    output_format_valid = output_validation["is_valid"]

    # Validate prediction formats
    generator_prediction_validation = validate_number_list_format(generator_prediction)
    generator_prediction_format_valid = generator_prediction_validation["is_valid"]

    guesser_prediction_validation = validate_number_list_format(guesser_prediction)
    guesser_prediction_format_valid = guesser_prediction_validation["is_valid"]

    # Normalize outputs for comparison if they're valid
    if code_executable and output_format_valid:
        normalized_actual = normalize_number_list_output(actual_output)
    else:
        normalized_actual = actual_output.strip()

    if generator_prediction_format_valid:
        normalized_generator_pred = normalize_number_list_output(generator_prediction)
    else:
        normalized_generator_pred = generator_prediction.strip()

    if guesser_prediction_format_valid:
        normalized_guesser_pred = normalize_number_list_output(guesser_prediction)
    else:
        normalized_guesser_pred = guesser_prediction.strip()

    # Check prediction correctness (using normalized comparison)
    generator_prediction_correct = (
        code_executable
        and output_format_valid
        and generator_prediction_format_valid
        and normalized_generator_pred == normalized_actual
    )

    guesser_prediction_correct = (
        code_executable
        and output_format_valid
        and guesser_prediction_format_valid
        and normalized_guesser_pred == normalized_actual
    )

    # REWARD STRUCTURE WITH NEW SPECIFICATION:
    # Generator penalties for execution/format failures:
    if not code_executable:
        generator_reward = -1.0  # Code doesn't run
    elif not output_format_valid:
        generator_reward = -1.0  # Code output wrong format
    elif not generator_prediction_format_valid:
        generator_reward = -1.0  # Generator prediction wrong format
    elif not generator_prediction_correct:
        generator_reward = -1.0  # Generator prediction incorrect
    else:
        # Code works AND generator predicts correctly - check guesser
        if guesser_prediction_correct:
            generator_reward = 0.2  # Both correct: +0.2
        else:
            generator_reward = 1.0  # Generator wins: +1.0

    # Guesser penalties:
    if not guesser_prediction_format_valid:
        guesser_reward = -1.0  # Guesser prediction wrong format
    elif not code_executable or not output_format_valid:
        guesser_reward = -1.0  # Can't win if generator's code/output is invalid
    else:
        # Formats are valid - check prediction correctness
        guesser_reward = 1.0 if guesser_prediction_correct else -1.0

    return {
        "generator": generator_reward,
        "guesser": guesser_reward,
        "generator_prediction_correct": generator_prediction_correct,
        "guesser_prediction_correct": guesser_prediction_correct,
        "output_format_valid": output_format_valid,
        "generator_prediction_format_valid": generator_prediction_format_valid,
        "guesser_prediction_format_valid": guesser_prediction_format_valid,
        "output_validation_error": output_validation.get("error_message", ""),
        "generator_prediction_validation_error": generator_prediction_validation.get(
            "error_message", ""
        ),
        "guesser_prediction_validation_error": guesser_prediction_validation.get(
            "error_message", ""
        ),
    }


# Set up model cache directory
cache_dir = config["generator_model"]["cache_dir"]
os.makedirs(cache_dir, exist_ok=True)

# Load generator model (trainable with GRPO)
generator_model_id = config["generator_model"]["id"]
print(f"📥 Loading generator model: {generator_model_id}")

generator_model = AutoModelForCausalLM.from_pretrained(
    generator_model_id,
    torch_dtype=torch.float16,  # Use fp16
    device_map="auto",
    cache_dir=cache_dir,
    local_files_only=offline_mode,
    load_in_8bit=True,  # 8-bit quantization
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
    torch_dtype=torch.float16,
    device_map="auto",  # or "cpu" for guesser
    cache_dir=cache_dir,
    local_files_only=offline_mode,
    load_in_8bit=True,  # 8-bit quantization
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

# Initialize vLLM integration if enabled
vllm_config = config.get("vllm", {})
if vllm_config.get("enabled", False):
    print("🚀 Initializing vLLM integration...")
    from utils.vllm_client import initialize_vllm_integration

    try:
        vllm_integration = initialize_vllm_integration(
            vllm_config, generator_model_id, offline_mode
        )

        if vllm_integration.initialize():
            print("✅ vLLM server started successfully")
        else:
            print("⚠️ vLLM server failed to start, falling back to HuggingFace")

    except Exception as e:
        print(f"❌ vLLM initialization failed: {e}")
        print("⚠️ Continuing with HuggingFace generation only")
else:
    print("🚫 vLLM disabled in configuration")

# Initialize ICL memory
latest_memory = ICLMemory(max_size=config["icl"]["memory_size"])
print(f"🧠 Initialized ICL memory with size {config['icl']['memory_size']}")

# Create game dataset for GRPO
game_prompt = """You are Player 1 in a code generation game. Your goals are to:
1. Write Python code that executes successfully without errors
2. Correctly predict what your code will output
3. Write code whose output Player 2 will struggle to predict correctly

IMPORTANT FORMAT REQUIREMENT:
Your code MUST output a list of numbers that is between 2 and 50 numbers long.
Examples of valid outputs:
- [1, 2, 3, 4, 5]
- [3.14, 2.71, 1.41]
- [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

Write a complete Python program that generates and prints a list of numbers.
The program should be executable and produce a list output in the required format.

Focus on writing clean, working code that produces an interesting or non-obvious list that you can predict but Player 2 cannot.

Examples of good strategies:
- Mathematical sequences (Fibonacci, primes, factorials)
- Calculations with specific numeric results
- List comprehensions with non-obvious patterns
- String-to-number conversions
- Date/time calculations that produce numbers

Format your response with the Python code AND your prediction:
```python
# Your code here - must print a list of 2-50 numbers
print([...])  # Your list here
```

<prediction>
[exact list your code will produce, e.g., [1, 2, 3, 4, 5]]
</prediction>

Write a Python program that outputs a list of numbers and predict its output:"""

dataset_size = config["dataset"]["size"]
dataset = Dataset.from_dict({"prompt": [game_prompt] * dataset_size})

# Training parameters
num_steps = config["training"]["num_steps"]
games_per_step = config["training"]["games_per_step"]

# Initialize GRPO trainer
training_args = config["training_args"]

# Adaptive settings based on GPU
if platform_info["gpu_type"] == "V100":
    training_args["per_device_train_batch_size"] = min(
        training_args["per_device_train_batch_size"], 2
    )
    training_args["gradient_accumulation_steps"] = max(
        training_args.get("gradient_accumulation_steps", 1), 4
    )
    print("🔧 Adjusted training args for V100 memory constraints")

# Fix model config path for GRPOTrainer if in offline mode
if offline_mode:
    # Point the model config to actual cached snapshot directory so GRPOTrainer can find tokenizer files locally
    cached_model_dirs = glob.glob(
        os.path.join(
            cache_dir,
            f"models--{generator_model_id.replace('/', '--')}",
            "snapshots",
            "*",
        )
    )
    if cached_model_dirs:
        generator_model.config._name_or_path = cached_model_dirs[0]
        print(f"🔧 Set generator model config path to: {cached_model_dirs[0]}")

grpo_config = GRPOConfig(**training_args)

print("🎮 Starting GRPO Code Game with ICL Memory Training")

# Initialize wandb run (only if W&B is enabled by user)
if WANDB_ENABLED:
    # Create human-readable timestamp: Jul31_2025_14h30m
    timestamp = datetime.datetime.now().strftime("%b%d_%Y_%Hh%Mm")

    # Use consistent project name (no timestamp) - all runs from this script go to same project
    project_name = config["wandb"]["project_name_prefix"]

    # Create readable run name with timestamp
    run_name = f"grpo-code-game-icl-{timestamp}"

    # Allow environment variable override for project name
    if os.environ.get("WANDB_PROJECT"):
        project_name = os.environ["WANDB_PROJECT"]
        print(f"🔧 Using WANDB_PROJECT override: {project_name}")

    wandb.init(
        project=project_name,
        name=run_name,
        config={**config, **seed_manager.get_seed_info()},
    )
    print(
        f"✅ Initialized W&B run: {wandb.run.name} (Project: {project_name}, Offline mode: {offline_mode})"
    )

# Run initial MBPP evaluation if enabled
if config["evaluation"].get("enabled_initial", True) and mbpp_evaluator.config.enabled:
    print("🧪 Running initial MBPP evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("initial")
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


# Create ICL-enhanced reward function that plays games to get rewards
def icl_enhanced_reward_function(completions, **kwargs):
    """
    ICL-enhanced reward function that plays games between generator and ICL guesser.
    This is called by GRPO for each batch of completions.
    """
    print(f"🎮 Playing {len(completions)} games for reward calculation...")

    rewards = []
    generator_wins = 0
    guesser_wins = 0

    # Store guesser predictions for debug output
    guesser_predictions = {}
    game_details = {}

    for i, completion in enumerate(completions):
        try:
            # Extract code and prediction from completion
            code = extract_code_from_response(completion)
            prediction = extract_prediction_from_response(completion)

            if not code:
                rewards.append(-1.0)
                guesser_wins += 1
                guesser_predictions[i] = "No code generated"
                game_details[i] = {
                    "code": code,
                    "generator_prediction": prediction,
                    "guesser_prediction": "No code generated",
                    "actual_output": "",
                    "execution_success": False,
                    "game_rewards": {"generator": -1.0, "guesser": -1.0},
                }
                continue

            # Execute the code
            execution_result = safe_execute_code(code, config["game"]["timeout"])
            actual_output = (
                execution_result["output"] if execution_result["success"] else ""
            )

            # Sample ICL opponent (80/20 mix)
            opponent = sample_opponent(
                generator_model, guesser_model, guesser_tokenizer, device, latest_memory
            )
            guesser_prediction = opponent.predict_output(code)
            guesser_predictions[i] = guesser_prediction

            # Calculate rewards
            game_rewards = calculate_rewards(
                execution_result, prediction, guesser_prediction, actual_output
            )
            reward = game_rewards["generator"]
            rewards.append(reward)

            # Store detailed game info for debugging
            game_details[i] = {
                "code": code,
                "generator_prediction": prediction,
                "guesser_prediction": guesser_prediction,
                "actual_output": actual_output,
                "execution_success": execution_result["success"],
                "execution_time": execution_result.get("execution_time", 0),
                "execution_error": execution_result.get("error", ""),
                "game_rewards": game_rewards,
            }

            # Track wins
            if reward > 0:
                generator_wins += 1

                # Add to pending wins for ICL memory update
                if (
                    execution_result["success"]
                    and game_rewards["generator_prediction_correct"]
                ):
                    if not hasattr(icl_enhanced_reward_function, "pending_wins"):
                        icl_enhanced_reward_function.pending_wins = []
                    icl_enhanced_reward_function.pending_wins.append(
                        {
                            "code": code,
                            "expected_output": actual_output,
                            "brief_rationale": "Generator won: code executed correctly and prediction was accurate",
                        }
                    )
            else:
                guesser_wins += 1

        except Exception as e:
            print(f"Error in game {i}: {e}")
            rewards.append(-1.0)
            guesser_wins += 1
            guesser_predictions[i] = f"Error: {str(e)}"
            game_details[i] = {
                "code": code if "code" in locals() else "",
                "generator_prediction": prediction if "prediction" in locals() else "",
                "guesser_prediction": f"Error: {str(e)}",
                "actual_output": "",
                "execution_success": False,
                "game_rewards": {"generator": -1.0, "guesser": -1.0},
            }

    # Store for debug access
    icl_enhanced_reward_function._last_guesser_predictions = guesser_predictions
    icl_enhanced_reward_function._last_game_details = game_details

    # Update global game counter
    if not hasattr(icl_enhanced_reward_function, "games_played"):
        icl_enhanced_reward_function.games_played = 0
    icl_enhanced_reward_function.games_played += len(completions)

    # Batch ICL memory updates
    if (
        hasattr(icl_enhanced_reward_function, "pending_wins")
        and icl_enhanced_reward_function.pending_wins
        and icl_enhanced_reward_function.games_played % REFRESH_EVERY == 0
    ):

        print(
            f"🧠 Updating ICL memory with {len(icl_enhanced_reward_function.pending_wins)} new winning examples..."
        )
        latest_memory.add_and_prune(
            icl_enhanced_reward_function.pending_wins, k=config["icl"]["memory_size"]
        )
        save_snapshot(latest_memory)
        icl_enhanced_reward_function.pending_wins = []  # Reset
        print(
            f"📸 Saved memory snapshot ({len(snapshot_buf)}/{SNAPSHOT_MAX} snapshots)"
        )

        # Show current ICL memory state
        print(f"🧠 Current ICL memory contains {len(latest_memory.examples)} examples:")
        for i, ex in enumerate(latest_memory.examples[-3:]):  # Show last 3 examples
            code_preview = ex.code[:100] + "..." if len(ex.code) > 100 else ex.code
            print(
                f"   Example {len(latest_memory.examples) - 3 + i + 1}: Code='{code_preview}' → Output='{ex.expected_output}'"
            )
        if len(latest_memory.examples) > 3:
            print(f"   ... and {len(latest_memory.examples) - 3} older examples")

    # Debug: Show detailed results for first few games
    debug_config = config.get("debug", {})
    show_detailed = debug_config.get("show_detailed_games", 2)
    max_code_chars = debug_config.get("max_code_chars", 500)
    show_full_responses = debug_config.get("show_full_responses", False)
    show_execution_details = debug_config.get("show_execution_details", True)

    if show_detailed > 0:
        print(f"\n{'='*80}")
        print(
            f"🔍 DETAILED GAME RESULTS (showing {min(show_detailed, len(completions))} games)"
        )
        print(f"{'='*80}")

        for i in range(min(show_detailed, len(completions))):
            completion = completions[i]
            reward = rewards[i]

            print(f"\n🎮 Game {i+1}/{len(completions)} - Reward: {reward:+.2f}")
            print("-" * 60)

            # Get detailed game info if available
            game_detail = game_details.get(i, {})

            if show_full_responses:
                print(f"📝 RL GENERATOR FULL RESPONSE:")
                print(f"```\n{completion}\n```")

            # Use stored details if available, otherwise extract
            code = game_detail.get("code", extract_code_from_response(completion))
            generator_prediction = game_detail.get(
                "generator_prediction", extract_prediction_from_response(completion)
            )
            guesser_prediction = game_detail.get("guesser_prediction", "Not available")
            actual_output = game_detail.get("actual_output", "")
            execution_success = game_detail.get("execution_success", False)
            execution_time = game_detail.get("execution_time", 0)
            execution_error = game_detail.get("execution_error", "")
            game_rewards_detail = game_detail.get("game_rewards", {})

            print(f"🤖 RL GENERATOR Generated Code ({len(code)} chars):")
            display_code = code[:max_code_chars] + (
                "..." if len(code) > max_code_chars else ""
            )
            print(f"```python\n{display_code}\n```")

            print(f"🎯 RL GENERATOR Prediction: '{generator_prediction}'")
            print(f"🤔 ICL GUESSER Prediction: '{guesser_prediction}'")

            if show_execution_details and code:
                print(f"🔧 Execution Results:")
                print(f"   Success: {'✅' if execution_success else '❌'}")
                print(f"   Runtime: {execution_time:.3f}s")

                if execution_success:
                    print(f"   Actual Output: '{actual_output}'")

                    # Show prediction correctness
                    generator_correct = game_rewards_detail.get(
                        "generator_prediction_correct", False
                    )
                    guesser_correct = game_rewards_detail.get(
                        "guesser_prediction_correct", False
                    )

                    print(
                        f"📊 Generator Prediction Correct: {'✅' if generator_correct else '❌'}"
                    )
                    print(
                        f"📊 Guesser Prediction Correct: {'✅' if guesser_correct else '❌'}"
                    )

                    # Show format validation results
                    output_format_valid = game_rewards_detail.get(
                        "output_format_valid", False
                    )
                    gen_format_valid = game_rewards_detail.get(
                        "generator_prediction_format_valid", False
                    )
                    guess_format_valid = game_rewards_detail.get(
                        "guesser_prediction_format_valid", False
                    )

                    print(
                        f"📏 Output Format Valid: {'✅' if output_format_valid else '❌'}"
                    )
                    print(
                        f"📏 Generator Format Valid: {'✅' if gen_format_valid else '❌'}"
                    )
                    print(
                        f"📏 Guesser Format Valid: {'✅' if guess_format_valid else '❌'}"
                    )

                    # Show winner
                    if generator_correct and not guesser_correct:
                        print(
                            f"🏆 Winner: Generator (correct prediction, guesser wrong)"
                        )
                    elif guesser_correct:
                        print(f"🏆 Winner: Guesser (correct prediction)")
                    else:
                        print(f"🏆 Winner: None (no correct predictions)")
                else:
                    error_msg = execution_error[:200] + (
                        "..." if len(execution_error) > 200 else ""
                    )
                    print(f"   Error: {error_msg}")
                    print(f"🏆 Winner: Guesser (generator code failed to execute)")

        print(f"{'='*80}")

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(
        f"🏆 Batch results: Generator {generator_wins}, Guesser {guesser_wins}, Avg reward: {avg_reward:.2f}"
    )

    # Always show a quick summary of guesser predictions vs actual outputs
    print(f"\n📊 Quick Game Summary:")
    for i in range(min(5, len(completions))):  # Show up to 5 games
        detail = game_details.get(i, {})
        actual = detail.get("actual_output", "")[:50]  # Truncate long outputs
        guesser_pred = detail.get("guesser_prediction", "")[:50]
        generator_pred = detail.get("generator_prediction", "")[:50]
        success = detail.get("execution_success", False)

        if success:
            generator_match = (
                "✅"
                if detail.get("game_rewards", {}).get(
                    "generator_prediction_correct", False
                )
                else "❌"
            )
            guesser_match = (
                "✅"
                if detail.get("game_rewards", {}).get(
                    "guesser_prediction_correct", False
                )
                else "❌"
            )
            print(
                f"   Game {i+1}: Actual='{actual}' | RL Generator='{generator_pred}' {generator_match} | ICL Guesser='{guesser_pred}' {guesser_match}"
            )
        else:
            print(
                f"   Game {i+1}: Code failed to execute | RL Generator='{generator_pred}' ❌ | ICL Guesser='{guesser_pred}' ❌"
            )
    if len(completions) > 5:
        print(f"   ... and {len(completions) - 5} more games")

    # Log batch metrics to W&B if enabled
    if WANDB_ENABLED and wandb.run:
        wandb.log(
            {
                "training/avg_reward": avg_reward,
                "training/generator_wins": generator_wins,
                "training/guesser_wins": guesser_wins,
                "training/icl_memory_size": len(latest_memory.examples),
                "training/kl_beta": config["training_args"].get("kl_beta", None),
                "step": getattr(icl_enhanced_reward_function, "games_played", 0),
            }
        )

    return rewards


# Replace the simple reward function with the ICL-enhanced one
grpo_trainer = GRPOTrainer(
    model=generator_model,
    reward_funcs=[icl_enhanced_reward_function],  # Use ICL-enhanced reward function
    args=grpo_config,
    train_dataset=dataset,
)

# Add interval evaluation callback
from transformers import TrainerCallback


class IntervalEvaluationCallback(TrainerCallback):
    def __init__(
        self, evaluator, model, tokenizer, config, wandb_enabled, seed_manager
    ):
        self.evaluator = evaluator
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.wandb_enabled = wandb_enabled
        self.seed_manager = seed_manager
        self.eval_interval = config["evaluation"].get("eval_interval_steps", None)

    def on_step_end(self, args, state, control, **kwargs):
        # DEBUG: Always print to see if callback is being called
        print(f"🔍 CALLBACK DEBUG: on_step_end called at step {state.global_step}")
        print(
            f"🔍 CALLBACK DEBUG: eval_interval={self.eval_interval}, evaluator.enabled={self.evaluator.config.enabled}"
        )
        print(
            f"🔍 CALLBACK DEBUG: condition check: step > 0: {state.global_step > 0}, step % interval == 0: {state.global_step % self.eval_interval == 0 if self.eval_interval else 'N/A'}"
        )

        # Run interval evaluation if enabled and it's time
        if (
            self.eval_interval
            and self.evaluator.config.enabled
            and state.global_step > 0  # Skip step 0 (initial eval already done)
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


# Add the callback to trainer
interval_callback = IntervalEvaluationCallback(
    mbpp_evaluator,
    generator_model,
    generator_tokenizer,
    config,
    WANDB_ENABLED,
    seed_manager,
)
grpo_trainer.add_callback(interval_callback)

print("🎮 Starting GRPO Code Game with ICL Memory Training")

# Run GRPO training - this will automatically call our ICL-enhanced reward function
print("🏋️ Starting GRPO training with ICL-enhanced rewards...")
grpo_trainer.train()

print("🏁 GRPO Code Game with ICL Memory training completed!")

# Run final MBPP evaluation if enabled
if config["evaluation"].get("enabled_final", True) and mbpp_evaluator.config.enabled:
    print("🧪 Running final MBPP evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("final")
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
        "examples": [
            {
                "code": ex.code,
                "expected_output": ex.expected_output,
                "brief_rationale": ex.brief_rationale,
            }
            for ex in latest_memory.examples
        ],
        "games_played": getattr(icl_enhanced_reward_function, "games_played", 0),
    }
    json.dump(memory_data, f, indent=2)

print(f"💾 Saved final checkpoint")

if WANDB_ENABLED and wandb.run:
    wandb.finish()
    print("✅ Training completed (W&B run finished)")
else:
    print("✅ Training completed (W&B logging was disabled)")
