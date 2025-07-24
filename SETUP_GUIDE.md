# ConnectX Qwen RL Project - Setup and Running Guide

## Overview
This project implements a Connect-X reinforcement learning agent using Qwen-2.5-3B model with PEFT-LoRA and PPO training. The agent learns to play Connect-X games through self-play and can be evaluated against various benchmarks.

## Prerequisites
- Python 3.11 or higher
- CUDA-compatible GPU (recommended for training)
- Git
- Weights & Biases account (for logging)

## Installation

### 1. Clone and Setup Environment
```bash
# Navigate to project directory
cd /home/ubuntu/game_project_rl

# Create virtual environment (recommended)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### 2. Verify Installation
```bash
# Run import tests
python -m pytest tests/test_imports.py -v
```

## Weights & Biases Setup

### 1. Create W&B Account
- Visit [wandb.ai](https://wandb.ai) and create an account
- Get your API key from your account settings

### 2. Login to W&B
```bash
# Login with your API key
wandb login

# Or set environment variable
export WANDB_API_KEY=your_api_key_here
```

### 3. Configure W&B Settings
Edit `configs/ppo.yaml` to customize logging:
```yaml
logging:
  project: "connectx-qwen-rl"          # Your W&B project name
  entity: "your_wandb_username"        # Your W&B username/team (optional)
  log_interval: 1                      # Log every N epochs
  save_interval: 5                     # Save checkpoints every N epochs
```

## Running the Project

### 1. Training the Agent
```bash
# Basic training with default config
python -m training.ppo_train --config configs/ppo.yaml

# Or using the console script (if installed with pip install -e .)
connectx-train --config configs/ppo.yaml

# Training with custom config
python -m training.ppo_train --config configs/test_ppo.yaml
```

### 2. Configuration Options
Key configuration files:
- `configs/ppo.yaml` - Main training configuration
- `configs/test_ppo.yaml` - Test/debugging configuration

Important configuration sections:
```yaml
model:
  name: "Qwen/Qwen2.5-3B"              # Base model
  lora:                                 # LoRA configuration
    r: 16                              # LoRA rank
    alpha: 32                          # LoRA alpha
    dropout: 0.1                       # LoRA dropout

training:
  learning_rate: 1.0e-5                # Learning rate
  batch_size: 32                       # Training batch size
  num_epochs: 10                       # Number of training epochs
  episodes_per_epoch: 20               # Episodes per epoch

ppo:
  gamma: 0.99                          # Discount factor
  clip_epsilon: 0.2                    # PPO clipping parameter
  value_loss_coef: 0.5                # Value loss coefficient
```

### 3. Evaluation
```bash
# Run evaluation
python evaluation/evalplus_runner.py

# Or using console script
connectx-eval
```

## Docker Usage

### 1. Build Docker Image
```bash
docker build -t connectx-qwen-rl .
```

### 2. Run Training in Docker
```bash
# Run with GPU support
docker run --gpus all -v $(pwd)/checkpoints:/workspace/checkpoints connectx-qwen-rl

# Run with W&B (pass API key)
docker run --gpus all -e WANDB_API_KEY=your_api_key \
  -v $(pwd)/checkpoints:/workspace/checkpoints connectx-qwen-rl
```

## Project Structure
```
game_project_rl/
├── agents/                    # Agent implementations
│   ├── base_agent.py         # Base agent interface
│   ├── qwen_policy.py        # Qwen-based policy
│   └── random_agent.py       # Random baseline agent
├── checkpoints/              # Model checkpoints
├── configs/                  # Configuration files
│   ├── ppo.yaml             # Main PPO config
│   └── test_ppo.yaml        # Test config
├── environment/              # Game environment
│   ├── connectx_wrapper.py  # Connect-X environment wrapper
│   └── reward_shaping.py    # Reward shaping utilities
├── evaluation/               # Evaluation scripts
│   ├── evalplus_runner.py   # EvalPlus benchmark runner
│   └── game_evaluator.py    # Game evaluation utilities
├── training/                 # Training pipeline
│   ├── ppo_train.py         # Main training script
│   ├── lora_utils.py        # LoRA utilities
│   └── wandb_logger.py      # W&B logging
└── tests/                   # Unit tests
```

## Monitoring and Logging

### W&B Dashboard
- Training progress: Real-time loss curves, rewards, and metrics
- Model performance: Win rates, game statistics
- Hyperparameter tracking: All config parameters logged
- Game visualizations: Board states and move sequences

### Local Checkpoints
Checkpoints are saved to `checkpoints/` directory:
- `best_model.pt` - Best performing model
- `latest_model.pt` - Most recent checkpoint
- `epoch_N.pt` - Periodic saves

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Enable gradient accumulation
   - Use smaller model variant

2. **W&B Login Issues**
   ```bash
   wandb login --relogin
   # Or set API key directly
   export WANDB_API_KEY=your_key
   ```

3. **Import Errors**
   ```bash
   # Reinstall in development mode
   pip install -e .
   ```

4. **Permission Errors (Docker)**
   ```bash
   # Fix checkpoint directory permissions
   sudo chown -R $USER:$USER checkpoints/
   ```

### Performance Tips

1. **GPU Utilization**
   - Monitor GPU usage with `nvidia-smi`
   - Adjust batch size for optimal GPU memory usage

2. **Training Speed**
   - Use mixed precision training (add to config)
   - Optimize data loading with multiple workers

3. **W&B Performance**
   - Set `WANDB_MODE=offline` for faster training
   - Sync later with `wandb sync`

## Environment Variables

Useful environment variables:
```bash
export WANDB_PROJECT=connectx-qwen-rl    # Default W&B project
export WANDB_ENTITY=your_username        # Default W&B entity
export CUDA_VISIBLE_DEVICES=0            # Specify GPU
export WANDB_MODE=offline                # Offline logging
export PYTHONPATH=/home/ubuntu/game_project_rl  # Python path
```

## Next Steps

1. **Custom Configurations**: Modify `configs/ppo.yaml` for your needs
2. **Hyperparameter Tuning**: Use W&B sweeps for optimization
3. **Model Variants**: Try different Qwen model sizes
4. **Environment Extensions**: Add new reward shaping or game variants
5. **Evaluation**: Implement custom benchmarks in `evaluation/`

## Support

For issues and questions:
- Check the troubleshooting section above
- Review W&B logs for training insights
- Examine checkpoint files for model state
- Run tests to verify installation: `python -m pytest tests/ -v`