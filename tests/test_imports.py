"""Compile-gate tests to ensure all modules import correctly."""

import pytest


def test_agent_imports():
    """Test that all agent modules import without errors."""
    from agents import BaseAgent, QwenPolicyAgent, RandomAgent
    from agents.base_agent import BaseAgent as BaseAgentDirect
    from agents.qwen_policy import QwenPolicyAgent as QwenPolicyAgentDirect
    from agents.random_agent import RandomAgent as RandomAgentDirect
    
    assert BaseAgent is not None
    assert QwenPolicyAgent is not None
    assert RandomAgent is not None


def test_environment_imports():
    """Test that environment modules import correctly."""
    from environment import ConnectXWrapper, RewardShaper
    from environment.connectx_wrapper import ConnectXWrapper as ConnectXWrapperDirect
    from environment.reward_shaping import RewardShaper as RewardShaperDirect
    
    assert ConnectXWrapper is not None
    assert RewardShaper is not None


def test_training_imports():
    """Test that training modules import correctly."""
    from training import setup_lora_model, WandBLogger
    from training.ppo_train import ConnectXPPOTrainer
    from training.lora_utils import setup_lora_model as setup_lora_model_direct
    from training.wandb_logger import WandBLogger as WandBLoggerDirect
    
    assert setup_lora_model is not None
    assert WandBLogger is not None
    assert ConnectXPPOTrainer is not None


def test_evaluation_imports():
    """Test that evaluation modules import correctly."""
    from evaluation import EvalPlusRunner, GameEvaluator
    from evaluation.evalplus_runner import EvalPlusRunner as EvalPlusRunnerDirect
    from evaluation.game_evaluator import GameEvaluator as GameEvaluatorDirect
    
    assert EvalPlusRunner is not None
    assert GameEvaluator is not None


def test_external_dependencies():
    """Test that all external dependencies are available."""
    import torch
    import transformers
    import peft
    import trl
    import kaggle_environments
    import wandb
    import numpy
    import pandas
    import matplotlib
    import seaborn
    import yaml
    import pytest
    
    # Test that key classes/functions are available
    assert hasattr(torch, 'cuda')
    assert hasattr(transformers, 'AutoTokenizer')
    assert hasattr(peft, 'LoraConfig')
    assert hasattr(trl, 'PPOTrainer')
    assert hasattr(kaggle_environments, 'make')


if __name__ == "__main__":
    pytest.main([__file__])