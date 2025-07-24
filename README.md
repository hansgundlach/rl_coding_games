# ConnectX-Qwen-RL

Connect-X RL agent using Qwen-3-4B with PEFT-LoRA and PPO training, evaluated on HumanEval-Plus and MBPP-Plus benchmarks.

## Features

- **Model**: Qwen-3-4B base model with LoRA fine-tuning
- **Training**: TRL PPO reinforcement learning on Kaggle Connect-X
- **Environment**: Kaggle Connect-X with custom reward shaping
- **Evaluation**: HumanEval-Plus and MBPP-Plus benchmarks via evalplus
- **Logging**: Weights & Biases integration with comprehensive metrics
- **Hardware**: Optimized for 1x A100 40GB on RunPod

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Unit Tests

```bash
pytest
```

### RL Training

```bash
python -m training.ppo_train --config configs/ppo.yaml
```

### Benchmark Evaluation

```bash
python evaluation/evalplus_runner.py checkpoints/latest
```

## Project Structure

```
connectx-qwen-rl/
├── agents/                 # Connect-X agents
│   ├── base_agent.py      # Abstract base agent
│   ├── qwen_policy.py     # Qwen-based policy agent
│   └── random_agent.py    # Random baseline agent
├── configs/               # Configuration files
│   ├── ppo.yaml          # PPO training config
│   └── model.yaml        # Model configuration
├── environment/           # Environment wrappers
│   ├── connectx_wrapper.py  # Gym wrapper for Connect-X
│   └── reward_shaping.py    # Custom reward functions
├── training/              # Training pipeline
│   ├── ppo_train.py      # Main PPO training script
│   ├── lora_utils.py     # LoRA utilities
│   └── wandb_logger.py   # W&B logging
├── evaluation/            # Evaluation utilities
│   ├── evalplus_runner.py   # EvalPlus benchmark runner
│   └── game_evaluator.py   # Connect-X game evaluation
├── tests/                 # Test suite
│   ├── test_imports.py   # Import compilation tests
│   ├── test_connectx.py  # Connect-X game logic tests
│   └── test_agents.py    # Agent functionality tests
└── checkpoints/          # Model checkpoints
```

## Training Configuration

Key training parameters in `configs/ppo.yaml`:

- **Model**: Qwen/Qwen2.5-3B with LoRA (r=16, α=32)
- **PPO**: Learning rate 1e-5, batch size 32, 4 epochs
- **Environment**: 6×7 Connect-X with reward shaping
- **Episodes**: 50 per epoch, 100 total epochs
- **Evaluation**: Pre/post training benchmarks with pass@10 logging

## Evaluation Benchmarks

The project evaluates model performance on:

1. **HumanEval-Plus**: Code generation benchmark with additional test cases
2. **MBPP-Plus**: Python programming problems with enhanced evaluation
3. **Connect-X Performance**: Win rate and game statistics

Results are logged to Weights & Biases under tags:
- `pre_rl`: Before reinforcement learning
- `post_rl`: After reinforcement learning

## Hardware Requirements

- **GPU**: NVIDIA A100 40GB (or equivalent)
- **CUDA**: 12.1+
- **Python**: 3.11+
- **Memory**: 40GB GPU memory for full model training

## Docker Support

```bash
docker build -t connectx-qwen-rl .
docker run --gpus all connectx-qwen-rl
```

## License

MIT License - see LICENSE file for details.