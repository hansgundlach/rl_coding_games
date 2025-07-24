from setuptools import setup, find_packages

setup(
    name="connectx-qwen-rl",
    version="0.1.0",
    description="Connect-X RL agent using Qwen-3-4B with PEFT-LoRA and PPO",
    author="ConnectX-Qwen-RL Team",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "peft>=0.6.0",
        "trl>=0.7.0",
        "kaggle-environments>=1.12.0",
        "wandb>=0.16.0",
        "evalplus>=0.2.0",
    ],
    entry_points={
        "console_scripts": [
            "connectx-train=training.ppo_train:main",
            "connectx-eval=evaluation.evalplus_runner:main",
        ],
    },
)