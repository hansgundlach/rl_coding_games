"""
Alignment Evaluation System

This module provides alignment evaluation capabilities for reinforcement learning
training of language models, using a battery of alignment-relevant prompts.

Key Components:
- evaluator: Core alignment evaluation functionality
- Configuration management for different training scenarios

Usage:
    from evaluation.alignment import AlignmentEvaluator, AlignmentEvalConfig
    
    config = AlignmentEvalConfig(enabled=True, eval_interval_steps=20)
    evaluator = AlignmentEvaluator(config)
    results = evaluator.evaluate_model(model, tokenizer)
"""

from .evaluator import AlignmentEvaluator, AlignmentEvalConfig, create_alignment_eval_config_for_training

__all__ = [
    'AlignmentEvaluator',
    'AlignmentEvalConfig', 
    'create_alignment_eval_config_for_training'
]

__version__ = "1.0.0"