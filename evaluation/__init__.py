"""
MBPP Evaluation System for RL Training

This package provides MBPP (Mostly Basic Python Problems) evaluation capabilities
for reinforcement learning training of code generation models.

Key Components:
- mbpp.evaluator: Core MBPP evaluation functionality
- configs.loader: Configuration management system
- datasets: Sample and full MBPP datasets

Usage:
    from evaluation import MBPPEvaluator, create_eval_config_for_training
    
    config = create_eval_config_for_training("grpo_code_game")
    evaluator = MBPPEvaluator(config)
    results = evaluator.evaluate_model(model, tokenizer)
"""

# Legacy evaluation utilities
from .evalplus_runner import EvalPlusRunner
from .game_evaluator import GameEvaluator

# New MBPP evaluation system
from evaluation.mbpp.evaluator import MBPPEvaluator, EvalConfig
from evaluation.configs.loader import create_eval_config_for_training, load_eval_config, print_config_summary

# Alignment evaluation system
from evaluation.alignment.evaluator import AlignmentEvaluator, AlignmentEvalConfig, create_alignment_eval_config_for_training

__all__ = [
    # Legacy
    "EvalPlusRunner", 
    "GameEvaluator",
    # New MBPP system
    'MBPPEvaluator',
    'EvalConfig', 
    'create_eval_config_for_training',
    'load_eval_config',
    'print_config_summary',
    # Alignment evaluation system
    'AlignmentEvaluator',
    'AlignmentEvalConfig',
    'create_alignment_eval_config_for_training'
]

__version__ = "1.0.0"