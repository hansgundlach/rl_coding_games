#!/usr/bin/env python3
"""
SPIRAL Code Generation Game - Self-Play Implementation

Game Rules:
- Player 1 (Generator): Generates Python code and knows the expected output
- Player 2 (Guesser): Must predict the output of the generated code
- Both players are the same model (self-play)

Rewards:
Player 1: +1 if code is executable AND Player 2 fails to guess output correctly, -1 otherwise
Player 2: +1 if successfully guesses the output, -1 if fails to guess correctly

Based on SPIRAL: Self-Play on Zero-Sum Games with Role-Conditioned Advantage Estimation
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
import subprocess
import tempfile
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
import yaml
import argparse


print("🚀 Starting SPIRAL Code Generation Game Training...")
print("📋 Initializing components...")

# Parse command line arguments
parser = argparse.ArgumentParser(description="SPIRAL Code Generation Game Training")
parser.add_argument(
    "--config",
    type=str,
    default="configs/spiral_code_game.yaml",
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
                sys.exit(1)
            current = current[k]

        # Check if final key exists (warn but don't fail - might be creating new key)
        final_key = keys[-1]
        if final_key not in current:
            print(f"⚠️ WARNING: Creating new config key '{key}'")
            print(f"   Available keys at this level: {list(current.keys())}")

        old_value = current.get(final_key, "NOT_SET")
        current[final_key] = value
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


@dataclass
class CodeGameTrajectory:
    """Store trajectory data for a single code generation game."""

    generator_data: Dict  # Player 1 (generates code + predicts output)
    guesser_data: Dict  # Player 2 (guesses output)
    game_outcome: Dict  # Final results and rewards
    execution_result: Dict  # Code execution results


class CodeGenerationGame:
    """
    Code Generation Game implementation for SPIRAL self-play.

    NEW Rules:
    - Player 1 generates code AND predicts its output
    - Player 2 must guess the output without seeing the actual execution
    - Player 1 wins if: code executable AND self-prediction correct AND Player 2 wrong
    - Player 2 wins if: prediction correct
    """

    def __init__(self, timeout: int = 3):
        self.timeout = timeout
        self.reset()

    def reset(self):
        """Reset the game to initial state."""
        self.game_over = False
        self.execution_result = None
        self.generator_code = None
        self.generator_prediction = None  # NEW: Player 1's prediction
        self.actual_output = None
        self.guesser_prediction = None

    def get_generator_prompt(self) -> str:
        """Get prompt for Player 1 (code generator + predictor)."""
        return """You are Player 1 in a code generation game. Your goals are to:
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

Focus on writing clean, working code that produces an interesting or non-obvious list that you can predict but Player 2 cannot. Player 2 only sees the code, not the output or your thoughts.

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

    def get_guesser_prompt(self, generator_code: str) -> str:
        """Get prompt for Player 2 (output guesser)."""
        return f"""You are Player 2 in a code generation game. Player 1 has written the following Python code:

```python
{generator_code}
```

Your goal is to predict EXACTLY what this code will output when executed.

IMPORTANT FORMAT REQUIREMENT:
The code should output a list of numbers that is between 2 and 50 numbers long.
Your prediction must also be in this format.

Examples of valid predictions:
- [1, 2, 3, 4, 5]
- [3.14, 2.71, 1.41]
- [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

Think step by step about what this code does:
1. Analyze each line of code
2. Trace through the execution
3. Determine the final list output
4. Count the numbers in your predicted list (must be 2-50)

Provide your prediction in this exact format:
<prediction>
[exact list here, e.g., [1, 2, 3, 4, 5]]
</prediction>

What list of numbers will this code output?"""

    def extract_code_from_response(self, response: str) -> str:
        """Extract Python code from model response."""
        # Look for code in markdown format
        code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Fallback: return the response as-is
        return response.strip()

    def extract_prediction_from_response(self, response: str) -> str:
        """Extract prediction from model response."""
        # Look for prediction tags
        pred_match = re.search(
            r"<prediction>\s*(.*?)\s*</prediction>", response, re.DOTALL
        )
        if pred_match:
            return pred_match.group(1).strip()

        # Fallback: return empty string if no prediction found
        return ""

    def play_game(self, model, tokenizer, device, step: int = 0) -> CodeGameTrajectory:
        """
        Play a single game between two players (same model with different roles).

        Returns complete trajectory data for training.
        """
        # Player 1: Generate code AND predict output
        generator_prompt = self.get_generator_prompt()

        inputs = tokenizer(
            generator_prompt, return_tensors="pt", truncation=True, max_length=512
        )
        if device.type == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Seed for deterministic generator response
        seed_manager.seed_for_generation(step=step, generation_idx=0)

        with torch.no_grad():
            generator_outputs = model.generate(
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
        self.generator_code = self.extract_code_from_response(generator_response)
        self.generator_prediction = self.extract_prediction_from_response(
            generator_response
        )

        # Execute the generated code to get actual output
        self.execution_result = safe_execute_code(self.generator_code, self.timeout)
        self.actual_output = (
            self.execution_result["output"] if self.execution_result["success"] else ""
        )

        # Player 2: Guess the output
        guesser_prompt = self.get_guesser_prompt(self.generator_code)

        inputs = tokenizer(
            guesser_prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        if device.type == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Seed for deterministic guesser response
        seed_manager.seed_for_generation(step=step, generation_idx=1)

        with torch.no_grad():
            guesser_outputs = model.generate(
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
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        guesser_response = tokenizer.decode(
            guesser_outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        # Extract prediction from guesser response
        self.guesser_prediction = self.extract_prediction_from_response(
            guesser_response
        )

        # Calculate rewards
        rewards = self.calculate_rewards()

        return CodeGameTrajectory(
            generator_data={
                "prompt": generator_prompt,
                "response": generator_response,
                "code": self.generator_code,
                "prediction": self.generator_prediction,  # NEW
                "role": "generator",
            },
            guesser_data={
                "prompt": guesser_prompt,
                "response": guesser_response,
                "prediction": self.guesser_prediction,
                "role": "guesser",
            },
            game_outcome={
                "generator_reward": rewards["generator"],
                "guesser_reward": rewards["guesser"],
                "code_executable": self.execution_result["success"],
                "generator_prediction_correct": rewards["generator_prediction_correct"],
                "guesser_prediction_correct": rewards["guesser_prediction_correct"],
                "actual_output": self.actual_output,
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
            execution_result=self.execution_result,
        )

    def calculate_rewards(self) -> Dict:
        """
        Calculate rewards for both players with format validation penalties.

        Player 1 (Generator): Gets positive reward if:
        - Code is executable AND
        - Output follows list format (2-50 numbers) AND
        - Their prediction is correct AND follows format AND
        - Player 2's prediction is wrong

        Player 2 (Guesser): Gets positive reward if:
        - Their prediction is correct AND follows format

        FORMAT PENALTIES:
        - Heavy penalty (-1.0) for wrong format
        - Code that doesn't produce valid list format gets -1.0
        - Predictions that don't follow list format get -1.0
        """
        # Check if code is executable
        code_executable = self.execution_result["success"]

        # Validate output format
        output_validation = (
            validate_number_list_format(self.actual_output)
            if code_executable
            else {"is_valid": False, "error_message": "Code not executable"}
        )
        output_format_valid = output_validation["is_valid"]

        # Validate prediction formats
        generator_prediction_validation = validate_number_list_format(
            self.generator_prediction
        )
        generator_prediction_format_valid = generator_prediction_validation["is_valid"]

        guesser_prediction_validation = validate_number_list_format(
            self.guesser_prediction
        )
        guesser_prediction_format_valid = guesser_prediction_validation["is_valid"]

        # Normalize outputs for comparison if they're valid
        if code_executable and output_format_valid:
            normalized_actual = normalize_number_list_output(self.actual_output)
        else:
            normalized_actual = self.actual_output.strip()

        if generator_prediction_format_valid:
            normalized_generator_pred = normalize_number_list_output(
                self.generator_prediction
            )
        else:
            normalized_generator_pred = self.generator_prediction.strip()

        if guesser_prediction_format_valid:
            normalized_guesser_pred = normalize_number_list_output(
                self.guesser_prediction
            )
        else:
            normalized_guesser_pred = self.guesser_prediction.strip()

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

        # REWARD STRUCTURE WITH FORMAT PENALTIES:
        # Generator penalties:
        if not code_executable:
            generator_reward = -1.0  # Code doesn't run
        elif not output_format_valid:
            generator_reward = -1.0  # Code output wrong format
        elif not generator_prediction_format_valid:
            generator_reward = -1.0  # Generator prediction wrong format
        else:
            # Code runs and formats are valid - check game outcome
            generator_reward = (
                1.0
                if (generator_prediction_correct and not guesser_prediction_correct)
                else -1.0
            )

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


class RoleConditionedAdvantageEstimation:
    """
    Role-conditioned Advantage Estimation (RAE) from SPIRAL paper.
    Maintains separate baselines for each player role to reduce variance.
    """

    def __init__(self, alpha: float = 0.95):
        self.alpha = alpha  # EMA decay rate
        self.baselines = {"generator": 0.0, "guesser": 0.0}
        self.update_counts = {"generator": 0, "guesser": 0}

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

    def compute_advantage(self, role: str, reward: float) -> float:
        """Compute advantage for player by subtracting role-specific baseline."""
        return reward - self.baselines[role]

    def get_stats(self) -> Dict:
        """Get baseline statistics for logging."""
        return {
            "baseline_generator": self.baselines["generator"],
            "baseline_guesser": self.baselines["guesser"],
            "updates_generator": self.update_counts["generator"],
            "updates_guesser": self.update_counts["guesser"],
        }


def compute_policy_gradient_loss(
    model,
    tokenizer,
    trajectories: List[CodeGameTrajectory],
    rae: RoleConditionedAdvantageEstimation,
    device,
) -> torch.Tensor:
    """
    Compute policy gradient loss using SPIRAL's RAE method with memory optimization.
    """
    total_loss = torch.zeros(1, device=device, requires_grad=True)
    num_updates = 0

    for trajectory in trajectories:
        # Process both generator and guesser
        for role in ["generator", "guesser"]:
            if role == "generator":
                player_data = trajectory.generator_data
                reward = trajectory.game_outcome["generator_reward"]
            else:
                player_data = trajectory.guesser_data
                reward = trajectory.game_outcome["guesser_reward"]

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
                response, return_tensors="pt", truncation=True, max_length=128
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
print(f"📥 Using model from config: {model_id}")

print(f"📥 Loading model for SPIRAL code game: {model_id}")

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

# Setup optimizer with specified hyperparameters
optimizer_name = config["training"].get("optimizer", "Adam").lower()
if optimizer_name == "adamw":
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(config["training"]["adam_beta1"], config["training"]["adam_beta2"]),
        weight_decay=config["training"]["weight_decay"],
    )
    print(
        f"🔧 Using AdamW optimizer with lr={config['training']['learning_rate']}, betas=({config['training']['adam_beta1']}, {config['training']['adam_beta2']}), weight_decay={config['training']['weight_decay']}"
    )
else:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=(
            config["training"].get("adam_beta1", 0.9),
            config["training"].get("adam_beta2", 0.999),
        ),
    )
    print(f"🔧 Using Adam optimizer with lr={config['training']['learning_rate']}")

# Training parameters
num_steps = config["training"]["num_steps"]

# Add debug configuration
debug_config = config.get("debug", {})
max_games_to_show = debug_config.get(
    "show_detailed_games", 2
)  # Show first N games per step
show_all_games = debug_config.get("show_all_games", False)  # Override to show all
max_code_chars = debug_config.get(
    "max_code_chars", 500
)  # Max characters of code to display
show_full_responses = debug_config.get(
    "show_full_responses", False
)  # Show full model responses
show_execution_details = debug_config.get(
    "show_execution_details", True
)  # Show execution details

# Adaptive batch size based on GPU memory
if platform_info["gpu_type"] == "V100":
    games_per_step = config["training"]["games_per_step_v100"]
    print(f"🔧 Using V100 memory-optimized settings: {games_per_step} games per step")
else:
    games_per_step = config["training"]["games_per_step_other"]
    print(f"🔧 Using standard memory settings: {games_per_step} games per step")

print(
    f"🎮 Starting SPIRAL code game training: {num_steps} steps, {games_per_step} games per step"
)

# Initialize the game
game = CodeGenerationGame(timeout=config["game"]["timeout"])

# Initialize wandb run (only if W&B is enabled by user)
if WANDB_ENABLED:
    # Create human-readable timestamp: Jul31_2025_14h30m
    timestamp = datetime.datetime.now().strftime("%b%d_%Y_%Hh%Mm")
    project_name = f"{config['wandb']['project_name_prefix']}-{timestamp}"
    wandb.init(project=project_name, config={**config, **seed_manager.get_seed_info()})
    print(
        f"✅ Initialized W&B run: {wandb.run.name} (Project: {project_name}, Offline mode: {offline_mode})"
    )

# Run initial MBPP evaluation if enabled
if config["evaluation"].get("enabled_initial", True) and mbpp_evaluator.config.enabled:
    print("🧪 Running initial MBPP evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("initial")
    initial_results = mbpp_evaluator.evaluate_model(
        model, tokenizer, step=0, phase="initial"
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
    print(f"\n🎯 Step {step + 1}/{num_steps}")

    # Collect trajectories through self-play games
    trajectories = []
    generator_wins = 0
    guesser_wins = 0
    code_executable_count = 0
    generator_correct_predictions = 0
    guesser_correct_predictions = 0

    # Format validation tracking
    valid_output_format_count = 0
    valid_generator_prediction_format_count = 0
    valid_guesser_prediction_format_count = 0
    format_violation_count = 0

    for game_idx in range(games_per_step):
        trajectory = game.play_game(model, tokenizer, device, step=step)
        trajectories.append(trajectory)

        # 🔍 DETAILED GAME LOGGING - Show generated code and results
        should_show_game = show_all_games or (game_idx < max_games_to_show)

        if should_show_game:
            print(f"\n{'='*80}")
            print(f"🎮 Game {game_idx + 1}/{games_per_step} - Step {step + 1}")
            print(f"{'='*80}")

            # Show full responses if requested
            if show_full_responses:
                print(f"📝 Generator Full Response:")
                print(f"```\n{trajectory.generator_data.get('response', 'N/A')}\n```")
                print(f"📝 Guesser Full Response:")
                print(f"```\n{trajectory.guesser_data.get('response', 'N/A')}\n```")

            # Show generated code (with length limit)
            code = trajectory.generator_data["code"]
            print(f"🤖 Generated Code ({len(code)} chars):")
            display_code = code[:max_code_chars] + (
                "..." if len(code) > max_code_chars else ""
            )
            print("```python")
            print(display_code)
            print("```")

            # Show execution results
            if show_execution_details:
                exec_result = trajectory.execution_result
                print(f"\n🔧 Execution Results:")
                print(f"   Success: {'✅' if exec_result['success'] else '❌'}")
                print(f"   Runtime: {exec_result['execution_time']:.3f}s")

                if exec_result["success"]:
                    print(
                        f"   Actual Output: '{trajectory.game_outcome['actual_output']}'"
                    )
                else:
                    error_msg = exec_result["error"][:200] + (
                        "..." if len(exec_result["error"]) > 200 else ""
                    )
                    print(f"   Error: {error_msg}")
                    if exec_result["timeout"]:
                        print(
                            f"   ⏰ Timed out after {exec_result['execution_time']:.1f}s"
                        )

            # Show both predictions and format validation
            print(f"\n🎯 Predictions:")
            print(
                f"   Generator predicted: '{trajectory.generator_data['prediction']}'"
            )
            print(
                f"   Generator correct: {'✅' if trajectory.game_outcome['generator_prediction_correct'] else '❌'}"
            )
            print(
                f"   Generator format valid: {'✅' if trajectory.game_outcome['generator_prediction_format_valid'] else '❌'}"
            )
            if not trajectory.game_outcome["generator_prediction_format_valid"]:
                print(
                    f"   Generator format error: {trajectory.game_outcome['generator_prediction_validation_error']}"
                )

            print(f"   Guesser predicted: '{trajectory.guesser_data['prediction']}'")
            print(
                f"   Guesser correct: {'✅' if trajectory.game_outcome['guesser_prediction_correct'] else '❌'}"
            )
            print(
                f"   Guesser format valid: {'✅' if trajectory.game_outcome['guesser_prediction_format_valid'] else '❌'}"
            )
            if not trajectory.game_outcome["guesser_prediction_format_valid"]:
                print(
                    f"   Guesser format error: {trajectory.game_outcome['guesser_prediction_validation_error']}"
                )

            print(
                f"   Output format valid: {'✅' if trajectory.game_outcome['output_format_valid'] else '❌'}"
            )
            if not trajectory.game_outcome["output_format_valid"]:
                print(
                    f"   Output format error: {trajectory.game_outcome['output_validation_error']}"
                )

            # Show rewards
            print(f"\n🏆 Rewards:")
            print(f"   Generator: {trajectory.game_outcome['generator_reward']:+.1f}")
            print(f"   Guesser: {trajectory.game_outcome['guesser_reward']:+.1f}")

            print(f"{'='*60}")
        else:
            # Brief summary for remaining games
            status = "✅" if trajectory.execution_result["success"] else "❌"
            pred_status = (
                "✅" if trajectory.game_outcome["guesser_prediction_correct"] else "❌"
            )
            format_status = (
                "✅"
                if (
                    trajectory.game_outcome["output_format_valid"]
                    and trajectory.game_outcome["generator_prediction_format_valid"]
                    and trajectory.game_outcome["guesser_prediction_format_valid"]
                )
                else "❌"
            )
            print(
                f"🎮 Game {game_idx + 1}: Code {status}, Format {format_status}, Prediction {pred_status}, Gen: {trajectory.game_outcome['generator_reward']:+.1f}, Guess: {trajectory.game_outcome['guesser_reward']:+.1f}"
            )

        # Track statistics
        if trajectory.game_outcome["generator_reward"] > 0:
            generator_wins += 1
        else:
            guesser_wins += 1

        if trajectory.game_outcome["code_executable"]:
            code_executable_count += 1

        if trajectory.game_outcome["generator_prediction_correct"]:
            generator_correct_predictions += 1

        if trajectory.game_outcome["guesser_prediction_correct"]:
            guesser_correct_predictions += 1

        # Track format validation statistics
        if trajectory.game_outcome["output_format_valid"]:
            valid_output_format_count += 1

        if trajectory.game_outcome["generator_prediction_format_valid"]:
            valid_generator_prediction_format_count += 1

        if trajectory.game_outcome["guesser_prediction_format_valid"]:
            valid_guesser_prediction_format_count += 1

        # Count games with any format violations
        if not (
            trajectory.game_outcome["output_format_valid"]
            and trajectory.game_outcome["generator_prediction_format_valid"]
            and trajectory.game_outcome["guesser_prediction_format_valid"]
        ):
            format_violation_count += 1

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
    executable_rate = code_executable_count / games_per_step
    generator_prediction_accuracy = generator_correct_predictions / games_per_step
    guesser_prediction_accuracy = guesser_correct_predictions / games_per_step

    # Format validation rates
    output_format_rate = valid_output_format_count / games_per_step
    generator_format_rate = valid_generator_prediction_format_count / games_per_step
    guesser_format_rate = valid_guesser_prediction_format_count / games_per_step
    format_violation_rate = format_violation_count / games_per_step

    stats = {
        "step": step,
        "loss": loss.item(),
        "generator_wins": generator_wins,
        "guesser_wins": guesser_wins,
        "executable_rate": executable_rate,
        "generator_prediction_accuracy": generator_prediction_accuracy,
        "guesser_prediction_accuracy": guesser_prediction_accuracy,
        "output_format_rate": output_format_rate,
        "generator_format_rate": generator_format_rate,
        "guesser_format_rate": guesser_format_rate,
        "format_violation_rate": format_violation_rate,
        **rae_stats,
    }

    print(f"📊 Loss: {loss.item():.4f}")
    print(f"🏆 Generator wins: {generator_wins}, Guesser wins: {guesser_wins}")
    print(f"💻 Executable code rate: {executable_rate:.2%}")
    print(f"🎯 Generator prediction accuracy: {generator_prediction_accuracy:.2%}")
    print(f"🎯 Guesser prediction accuracy: {guesser_prediction_accuracy:.2%}")
    print(
        f"📋 Format compliance - Output: {output_format_rate:.2%}, Gen pred: {generator_format_rate:.2%}, Guess pred: {guesser_format_rate:.2%}"
    )
    print(f"⚠️ Format violations: {format_violation_rate:.2%}")
    print(
        f"📈 Baselines - Gen: {rae_stats['baseline_generator']:.3f}, Guess: {rae_stats['baseline_guesser']:.3f}"
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
        # Seed for consistent evaluation
        seed_manager.seed_for_evaluation_auto(f"interval_step_{step + 1}")
        interval_results = mbpp_evaluator.evaluate_model(
            model, tokenizer, step=step + 1, phase="interval"
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
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Save RAE state
        with open(f"{checkpoint_dir}/rae_state.json", "w") as f:
            json.dump(rae_stats, f, indent=2)

        print(f"💾 Saved checkpoint at step {step + 1}")

print("🏁 SPIRAL code game training completed!")

# Run final MBPP evaluation if enabled
if config["evaluation"].get("enabled_final", True) and mbpp_evaluator.config.enabled:
    print("🧪 Running final MBPP evaluation...")
    # Seed for consistent evaluation
    seed_manager.seed_for_evaluation_auto("final")
    final_results = mbpp_evaluator.evaluate_model(
        model, tokenizer, step=num_steps, phase="final"
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
