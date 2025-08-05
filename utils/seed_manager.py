#!/usr/bin/env python3
"""
Comprehensive Seed Management Utility

This module provides a unified approach to managing randomness across all
machine learning components to ensure reproducible experiments.

Features:
- Automatic seeding of Python random, NumPy, PyTorch, and transformers
- Per-process seed generation for distributed training
- Per-step seed derivation for consistent sampling
- MBPP evaluation problem selection seeding
- Environment variable override support

Usage:
    from utils.seed_manager import SeedManager
    
    # Basic usage with global seed
    seed_manager = SeedManager(base_seed=42)
    
    # Initialize all random number generators
    seed_manager.seed_everything()
    
    # Get deterministic seed for specific process/step
    step_seed = seed_manager.get_step_seed(step=5)
    eval_seed = seed_manager.get_evaluation_seed()
"""

import os
import random
import numpy as np
import torch
from typing import Optional, Dict, Any
import hashlib


class SeedManager:
    """
    Comprehensive seed management for reproducible ML experiments.
    
    Handles seeding of all random number generators used in the codebase:
    - Python's random module
    - NumPy random functions
    - PyTorch random functions (CPU and CUDA)
    - Transformers library sampling
    - Custom evaluation seeds
    """
    
    def __init__(self, base_seed: Optional[int] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the seed manager.
        
        Args:
            base_seed: Base seed for all random number generation. If None, will try to get from
                      environment variable SEED, then from config, then default to 42.
            config: Configuration dictionary that may contain seed settings.
        """
        # Determine base seed from multiple sources
        self.base_seed = self._determine_base_seed(base_seed, config)
        
        # Store configuration for later use
        self.config = config or {}
        
        # Track if we've already seeded (avoid double-seeding)
        self._seeded = False
        
        # Store seeds for logging and reproducibility
        self.seed_log = {
            "base_seed": self.base_seed,
            "python_random": None,
            "numpy": None,
            "torch_cpu": None,
            "torch_cuda": None,
            "transformers": None,
        }
        
        print(f"🎲 SeedManager initialized with base_seed={self.base_seed}")
    
    def _determine_base_seed(self, base_seed: Optional[int], config: Optional[Dict[str, Any]]) -> int:
        """Determine base seed from multiple sources in order of priority."""
        # 1. Explicit parameter
        if base_seed is not None:
            return base_seed
        
        # 2. Environment variable
        env_seed = os.environ.get("SEED")
        if env_seed is not None:
            try:
                return int(env_seed)
            except ValueError:
                print(f"⚠️ Invalid SEED environment variable '{env_seed}', ignoring")
        
        # 3. Config file
        if config and "seed" in config:
            return config["seed"]
        
        if config and "training" in config and "seed" in config["training"]:
            return config["training"]["seed"]
        
        # 4. Default fallback
        return 42
    
    def seed_everything(self, rank: int = 0) -> None:
        """
        Seed all random number generators for reproducible experiments.
        
        Args:
            rank: Process rank for distributed training (affects CUDA seeding)
        """
        if self._seeded:
            print("⚠️ SeedManager: Already seeded, skipping duplicate seeding")
            return
        
        print(f"🌱 Seeding all random number generators (rank={rank})...")
        
        # 1. Python's random module
        python_seed = self._derive_seed("python", rank)
        random.seed(python_seed)
        self.seed_log["python_random"] = python_seed
        
        # 2. NumPy random functions
        numpy_seed = self._derive_seed("numpy", rank)
        np.random.seed(numpy_seed)
        self.seed_log["numpy"] = numpy_seed
        
        # 3. PyTorch CPU random functions
        torch_cpu_seed = self._derive_seed("torch_cpu", rank)
        torch.manual_seed(torch_cpu_seed)
        self.seed_log["torch_cpu"] = torch_cpu_seed
        
        # 4. PyTorch CUDA random functions (if available)
        if torch.cuda.is_available():
            torch_cuda_seed = self._derive_seed("torch_cuda", rank)
            torch.cuda.manual_seed_all(torch_cuda_seed)
            self.seed_log["torch_cuda"] = torch_cuda_seed
            
            # For reproducible CUDA operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # 5. Transformers library (affects model generation)
        transformers_seed = self._derive_seed("transformers", rank)
        self.seed_log["transformers"] = transformers_seed
        
        # Set environment variable for transformers library to use
        os.environ["TRANSFORMERS_SEED"] = str(transformers_seed)
        
        self._seeded = True
        
        print("✅ All random number generators seeded successfully")
        self._print_seed_summary()
    
    def _derive_seed(self, component: str, rank: int = 0) -> int:
        """
        Derive a component-specific seed from the base seed.
        
        This ensures different components use different but deterministic seeds.
        """
        # Create a unique string for this component and rank
        seed_string = f"{self.base_seed}_{component}_{rank}"
        
        # Use hash to generate a deterministic but different seed
        hash_object = hashlib.md5(seed_string.encode())
        
        # Convert to integer in valid range
        return int(hash_object.hexdigest(), 16) % (2**31 - 1)
    
    def get_step_seed(self, step: int, component: str = "step") -> int:
        """
        Get a deterministic seed for a specific training step.
        
        This allows for consistent sampling within each step while varying
        between steps for diversity.
        
        Args:
            step: Training step number
            component: Component identifier (for multiple random operations per step)
        
        Returns:
            Deterministic seed for this step
        """
        return self._derive_seed(f"{component}_{step}")
    
    def get_evaluation_seed(self, phase: str = "eval") -> int:
        """
        Get a deterministic seed for evaluation phases.
        
        This ensures consistent problem selection in MBPP evaluations.
        
        Args:
            phase: Evaluation phase ("initial", "interval", "final", or custom)
        
        Returns:
            Deterministic seed for evaluation
        """
        return self._derive_seed(f"evaluation_{phase}")
    
    def get_process_seed(self, rank: int, world_size: int) -> int:
        """
        Get a process-specific seed for distributed training.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
        
        Returns:
            Process-specific seed
        """
        return self._derive_seed(f"process_{rank}_{world_size}")
    
    def seed_for_generation(self, step: int, generation_idx: int = 0) -> None:
        """
        Seed random number generators for consistent model generation.
        
        This should be called before each model.generate() call that needs
        to be reproducible.
        
        Args:
            step: Current training step
            generation_idx: Index of generation within the step (for multiple generations)
        """
        gen_seed = self.get_step_seed(step, f"generation_{generation_idx}")
        
        # Seed the generators that affect model sampling
        torch.manual_seed(gen_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(gen_seed)
        
        # Also seed numpy in case it's used in sampling
        np.random.seed(gen_seed % (2**31 - 1))
    
    def seed_for_evaluation(self, phase: str = "eval", problem_idx: Optional[int] = None, 
                           consistent_questions: bool = True) -> None:
        """
        Seed random number generators for consistent evaluation.
        
        Args:
            phase: Evaluation phase
            problem_idx: Specific problem index (for per-problem seeding)
            consistent_questions: If True, all interval evaluations use same questions.
                                If False, each step gets different questions.
        """
        if problem_idx is not None:
            eval_seed = self._derive_seed(f"evaluation_{phase}_problem_{problem_idx}")
        else:
            if consistent_questions and phase.startswith("interval_step_"):
                # Use consistent seed for all interval evaluations
                eval_seed = self.get_evaluation_seed("interval_consistent")
            else:
                eval_seed = self.get_evaluation_seed(phase)
        
        # Seed generators used in evaluation
        random.seed(eval_seed)
        np.random.seed(eval_seed % (2**31 - 1))
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
    
    def get_dataset_seed(self, dataset_name: str) -> int:
        """Get a seed for dataset operations (shuffling, sampling, etc.)."""
        return self._derive_seed(f"dataset_{dataset_name}")
    
    def get_eval_consistency_setting(self) -> bool:
        """
        Get whether to use consistent evaluation questions across all steps.
        
        Returns:
            True if all interval evaluations should use same questions (default),
            False if each step should get different questions.
        """
        # Check config first
        if self.config and "evaluation" in self.config:
            return self.config["evaluation"].get("consistent_questions", True)
        
        # Check environment variable override
        env_consistent = os.environ.get("EVAL_CONSISTENT_QUESTIONS", "true").lower()
        return env_consistent in ("true", "1", "yes", "on")
    
    def seed_for_evaluation_auto(self, phase: str = "eval", problem_idx: Optional[int] = None) -> None:
        """
        Seed evaluation with automatic consistency setting from config.
        
        This is a convenience method that uses the config/environment setting
        for consistent_questions automatically.
        """
        consistent = self.get_eval_consistency_setting()
        self.seed_for_evaluation(phase, problem_idx, consistent)
    
    def _print_seed_summary(self) -> None:
        """Print a summary of all seeds for logging."""
        print("\n" + "=" * 50)
        print("🎲 SEED SUMMARY")
        print("=" * 50)
        for component, seed in self.seed_log.items():
            if seed is not None:
                print(f"   {component}: {seed}")
        print("=" * 50 + "\n")
    
    def get_seed_info(self) -> Dict[str, Any]:
        """
        Get complete seed information for logging to wandb or saving.
        
        Returns:
            Dictionary with all seed information
        """
        return {
            "seed_manager": {
                "base_seed": self.base_seed,
                "seeded": self._seeded,
                "seed_log": self.seed_log.copy(),
            }
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SeedManager":
        """
        Create SeedManager from configuration dictionary.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Configured SeedManager instance
        """
        return cls(config=config)


def apply_seeds_to_generation_config(seed_manager: SeedManager, step: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply deterministic seeds to model generation configuration.
    
    This modifies generation parameters to use deterministic sampling when
    reproducibility is desired.
    
    Args:
        seed_manager: SeedManager instance
        step: Current training step
        config: Generation configuration dictionary
    
    Returns:
        Modified generation configuration
    """
    # Make a copy to avoid modifying the original
    gen_config = config.copy()
    
    # If temperature is > 0, we're doing sampling - seed the generators
    if gen_config.get("temperature", 0.0) > 0.0 or gen_config.get("do_sample", False):
        # Seed before generation
        seed_manager.seed_for_generation(step)
        
        # Optionally make sampling more deterministic
        if gen_config.get("make_deterministic", False):
            gen_config["temperature"] = 0.1  # Low but not zero
            gen_config["top_p"] = 0.9
            gen_config["do_sample"] = True
    
    return gen_config