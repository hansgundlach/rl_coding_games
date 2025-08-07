#!/usr/bin/env python3
"""
Configuration loader for MBPP evaluation system.
Supports YAML configuration files and environment variable overrides.
"""


# sbatch submit_spiral_code_game.sh --training.num_steps=200 --evaluation.eval_interval_steps=20 evaluation.num_questions=30

import os
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict
from evaluation.mbpp.evaluator import EvalConfig

logger = logging.getLogger(__name__)


def load_eval_config(config_path: Optional[str] = None, **overrides) -> EvalConfig:
    """
    Load evaluation configuration from YAML file with optional overrides.

    Args:
        config_path: Path to YAML config file. If None, searches for default configs.
        **overrides: Keyword arguments to override config values.

    Returns:
        EvalConfig instance with loaded/overridden values.
    """
    # Default configuration
    config_dict = asdict(EvalConfig())

    # Load from YAML if available
    yaml_config = load_yaml_config(config_path)
    if yaml_config:
        config_dict.update(yaml_config)

    # Apply environment variable overrides
    env_overrides = load_env_overrides()
    config_dict.update(env_overrides)

    # Apply direct overrides
    config_dict.update(overrides)

    # Create and return EvalConfig
    try:
        return EvalConfig(**config_dict)
    except TypeError as e:
        logger.error(f"Invalid configuration parameters: {e}")
        logger.info("Using default configuration")
        return EvalConfig()


def load_yaml_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path:
        # Search for default config files
        possible_paths = [
            "./configs/eval_config.yaml",
            "./eval_config.yaml",
            "../configs/eval_config.yaml",
            os.path.expanduser("~/.mbpp_eval_config.yaml"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break

    if not config_path or not os.path.exists(config_path):
        logger.info("No YAML config file found, using defaults")
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config or {}

    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def load_env_overrides() -> Dict[str, Any]:
    """Load configuration overrides from environment variables."""
    overrides = {}

    # Environment variable mappings
    env_mappings = {
        "MBPP_EVAL_ENABLED": ("enabled", lambda x: x.lower() in ["true", "1", "yes"]),
        "MBPP_EVAL_NUM_QUESTIONS": ("num_questions", int),
        "MBPP_EVAL_AT_START": (
            "eval_at_start",
            lambda x: x.lower() in ["true", "1", "yes"],
        ),
        "MBPP_EVAL_AT_END": (
            "eval_at_end",
            lambda x: x.lower() in ["true", "1", "yes"],
        ),
        "MBPP_EVAL_INTERVAL_STEPS": (
            "eval_interval_steps",
            lambda x: int(x) if x.lower() != "null" else None,
        ),
        "MBPP_DATASET_PATH": ("dataset_path", str),
        "MBPP_USE_SANITIZED": (
            "use_sanitized",
            lambda x: x.lower() in ["true", "1", "yes"],
        ),
        "MBPP_TEST_SPLIT": ("test_split", str),
        "MBPP_TIMEOUT_SECONDS": ("timeout_seconds", int),
        "MBPP_MAX_MEMORY_MB": ("max_memory_mb", int),
        "MBPP_SAFE_EXECUTION": (
            "safe_execution",
            lambda x: x.lower() in ["true", "1", "yes"],
        ),
        "MBPP_MAX_NEW_TOKENS": ("max_new_tokens", int),
        "MBPP_TEMPERATURE": ("temperature", float),
        "MBPP_DO_SAMPLE": ("do_sample", lambda x: x.lower() in ["true", "1", "yes"]),
        "MBPP_SAVE_RESULTS": (
            "save_results",
            lambda x: x.lower() in ["true", "1", "yes"],
        ),
        "MBPP_RESULTS_DIR": ("results_dir", str),
        "MBPP_VERBOSE": ("verbose", lambda x: x.lower() in ["true", "1", "yes"]),
    }

    for env_var, (config_key, converter) in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            try:
                overrides[config_key] = converter(env_value)
                logger.info(
                    f"Override from {env_var}: {config_key}={overrides[config_key]}"
                )
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid value for {env_var}={env_value}: {e}")

    return overrides


def get_platform_specific_config() -> Dict[str, Any]:
    """Get platform-specific configuration adjustments."""
    import socket

    hostname = os.uname().nodename.lower()
    current_path = os.getcwd().lower()

    # Detect platform
    is_supercloud = (
        "gridsan" in hostname
        or "supercloud" in hostname
        or "/home/gridsan/" in current_path
    )
    is_lambda = "lambda" in hostname or "/lambda/" in current_path

    config_adjustments = {}

    if is_supercloud:
        # MIT Supercloud V100 - conservative settings
        config_adjustments.update(
            {
                "num_questions": 5,  # Smaller for memory constraints
                "timeout_seconds": 5,  # Shorter timeout
                "max_new_tokens": 256,  # Shorter generations
                "results_dir": f"./eval_results_{hostname}",  # Unique per node
            }
        )
        logger.info("Applied Supercloud V100 configuration adjustments")

    elif is_lambda:
        # Lambda A100 - more aggressive settings
        config_adjustments.update(
            {
                "num_questions": 20,  # Can handle more problems
                "timeout_seconds": 15,  # More generous timeout
                "max_new_tokens": 768,  # Longer generations
            }
        )
        logger.info("Applied Lambda A100 configuration adjustments")

    # Check for internet connectivity for dataset path detection
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        internet_available = True
    except (socket.error, socket.timeout):
        internet_available = False

    if not internet_available and not config_adjustments.get("dataset_path"):
        # Offline environment - try to find local dataset
        possible_paths = [
            "./data/mbpp/sanitized-mbpp.json",
            "../data/mbpp/sanitized-mbpp.json",
            "/home/gridsan/shared/mbpp/sanitized-mbpp.json",  # Common shared location
        ]

        for path in possible_paths:
            if os.path.exists(path):
                config_adjustments["dataset_path"] = path
                logger.info(f"Auto-detected offline dataset at {path}")
                break

    return config_adjustments


def create_eval_config_for_training(training_script: str, **overrides) -> EvalConfig:
    """
    Create evaluation configuration specifically for training scripts.

    Args:
        training_script: Name of training script ("grpo_code_game" or "grpo_code_execution")
        **overrides: Additional configuration overrides
    """
    # Base configuration
    config_overrides = {}

    # Script-specific settings
    if training_script == "grpo_code_game":
        config_overrides.update(
            {
                "results_dir": "./eval_results/grpo_code_game",
                "num_questions": 10,  # Good balance for adversarial training
            }
        )
    elif training_script == "grpo_code_execution":
        config_overrides.update(
            {
                "results_dir": "./eval_results/grpo_code_execution",
                "num_questions": 10,  # Good balance for execution training
            }
        )
    elif training_script == "spiral_prisoners_dilemma":
        config_overrides.update(
            {
                "results_dir": "./eval_results/spiral_prisoners_dilemma",
                "num_questions": 10,  # Good balance for strategy training
            }
        )

    # Add platform-specific adjustments
    platform_config = get_platform_specific_config()
    config_overrides.update(platform_config)

    # Add user overrides
    config_overrides.update(overrides)

    # Load final configuration
    return load_eval_config(**config_overrides)


def print_config_summary(config: EvalConfig):
    """Print a summary of the evaluation configuration."""
    print("\n" + "=" * 50)
    print("📊 MBPP Evaluation Configuration")
    print("=" * 50)
    print(f"Enabled: {'✅' if config.enabled else '❌'}")

    if config.enabled:
        print(f"Questions: {config.num_questions}")
        print(f"Eval at start: {'✅' if config.eval_at_start else '❌'}")
        print(f"Eval at end: {'✅' if config.eval_at_end else '❌'}")

        if config.eval_interval_steps:
            print(f"Eval interval: every {config.eval_interval_steps} steps")
        else:
            print("Eval interval: none (only start/end)")

        print(f"Dataset: {config.dataset_path or 'auto-detect'}")
        print(f"Results dir: {config.results_dir}")
        print(f"Temperature: {config.temperature}")
        print(f"Max tokens: {config.max_new_tokens}")
        print(f"Timeout: {config.timeout_seconds}s")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing configuration loader...")

    # Test default config
    config = load_eval_config()
    print_config_summary(config)

    # Test with overrides
    config = load_eval_config(num_questions=5, temperature=0.1)
    print("With overrides:")
    print_config_summary(config)

    # Test training-specific config
    config = create_eval_config_for_training("grpo_code_game")
    print("Training-specific config:")
    print_config_summary(config)
