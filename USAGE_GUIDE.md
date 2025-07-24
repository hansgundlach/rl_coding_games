# Qwen ConnectX RL Training Pipeline Usage Guide

## Overview

This pipeline trains a Qwen language model to play Connect-X using Proximal Policy Optimization (PPO) with LoRA fine-tuning, and evaluates the model's coding performance before and after training.

## Quick Start

1. **Test the pipeline**:
   ```bash
   python test_pipeline.py
   ```

2. **Start Qwen PPO training**:
   ```bash
   python training/qwen_ppo_train.py --config configs/qwen_ppo.yaml
   ```

3. **Run original PPO training** (for comparison):
   ```bash
   python training/ppo_train.py --config configs/ppo.yaml
   ```

## Configuration

### Qwen PPO Configuration (`configs/qwen_ppo.yaml`)

Key settings you may want to adjust:

- **Model settings**:
  - `qwen.model_name`: Qwen model to use (default: "Qwen/Qwen2.5-3B")
  - `qwen.device`: Device for inference ("auto", "cuda", "cpu")
  - `qwen.temperature`: Sampling temperature for generation

- **LoRA settings**:
  - `lora.r`: LoRA rank (higher = more parameters)
  - `lora.lora_alpha`: LoRA scaling factor
  - `lora.target_modules`: Which modules to apply LoRA to

- **Training settings**:
  - `training.num_epochs`: Number of training epochs
  - `training.episodes_per_epoch`: Episodes per epoch
  - `training.learning_rate`: Learning rate for LoRA parameters

## Training Process

### What Happens During Training

1. **Model Loading**: Loads Qwen model with LoRA adapters
2. **Initial Evaluation**: Runs coding benchmarks (HumanEval-Plus, MBPP-Plus)
3. **Game Episodes**: Model plays Connect-X games against random opponent
4. **PPO Updates**: Updates LoRA parameters based on game rewards
5. **Periodic Evaluation**: Re-evaluates coding performance
6. **Checkpointing**: Saves LoRA weights periodically

### Expected Output

```
Loading Qwen model: Qwen/Qwen2.5-3B
Available GPU memory: 42.4 GB
Qwen model loaded successfully
trainable parameters: 41,943,040 || all parameters: 3,089,031,168 || trainable%: 1.36
Starting Qwen PPO training...
Running coding benchmark evaluation...

Epoch 1/20
  Episode 5: Reward=5.20, Length=15
  Episode 10: Reward=-2.40, Length=22
  ...
  Mean reward: 1.85 ± 3.22
  Policy loss: 0.1234

  Saved LoRA checkpoint: checkpoints/qwen_ppo/qwen_ppo_epoch_5
  Coding evaluation results: {'humaneval_plus': {'pass@1': 0.15, 'pass@10': 0.28}, ...}
```

## Evaluation

The pipeline evaluates the model on two types of tasks:

### 1. Connect-X Game Performance
- Win rate against random opponent
- Average episode length
- Reward progression over training

### 2. Coding Benchmarks
- **HumanEval-Plus**: Programming problems with additional test cases
- **MBPP-Plus**: Mostly Basic Python Problems with enhanced tests
- Metrics: pass@1 and pass@10 scores

## Monitoring

Training progress is logged to Weights & Biases with:
- Game performance metrics
- LoRA training loss
- Coding evaluation results
- Resource usage

View your runs at: https://wandb.ai/your-entity/connectx-qwen-rl

## Checkpoints

### Checkpoint Structure
```
checkpoints/qwen_ppo/
├── qwen_ppo_epoch_5/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
├── qwen_ppo_epoch_10/
└── ...
```

### Loading Checkpoints

```python
from agents.qwen_policy import QwenPolicyAgent

# Load agent with trained LoRA weights
agent = QwenPolicyAgent(
    model_name="Qwen/Qwen2.5-3B",
    lora_checkpoint="checkpoints/qwen_ppo/qwen_ppo_epoch_10"
)
```

## Evaluation Only

To evaluate a checkpoint without training:

```python
from evaluation.evalplus_runner import EvalPlusRunner

runner = EvalPlusRunner(
    model_path="Qwen/Qwen2.5-3B",
    lora_path="checkpoints/qwen_ppo/qwen_ppo_epoch_10"
)

results = runner.run_evaluation()
print(results)
```

## Troubleshooting

### Common Issues

1. **GPU Out of Memory**:
   - Reduce `training.batch_size`
   - Use smaller model (e.g., "Qwen/Qwen2.5-1.5B")
   - Set `qwen.device: "cpu"` for CPU training

2. **Model Loading Fails**:
   - Check internet connection for model download
   - Verify model name is correct
   - Clear model cache: `rm -rf ./model_cache`

3. **EvalPlus Installation Issues**:
   ```bash
   pip install evalplus --upgrade
   ```

4. **Slow Training**:
   - Reduce `training.episodes_per_epoch`
   - Increase `checkpoints.save_interval`
   - Disable coding evaluation: comment out evaluation calls

### Resource Requirements

- **Minimum**: 8GB GPU memory, 16GB RAM
- **Recommended**: 16GB+ GPU memory, 32GB+ RAM
- **Storage**: 10GB for model cache and checkpoints

## Advanced Usage

### Custom Reward Shaping

Modify `environment/reward_shaping.py` to customize game rewards:

```python
def compute_reward(self, board, action, done, winner):
    # Add your custom reward logic
    reward = 0
    if done:
        reward = 100 if winner == self.player else -100
    # Add intermediate rewards...
    return reward
```

### Custom LoRA Configuration

Experiment with different LoRA settings in `configs/qwen_ppo.yaml`:

```yaml
lora:
  r: 32  # Higher rank for more capacity
  lora_alpha: 64
  target_modules:
    - "q_proj"
    - "k_proj" 
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj" 
    - "down_proj"
    - "embed_tokens"  # Include embedding layer
```

### Multi-GPU Training

For multi-GPU setups, set device mapping:

```yaml
qwen:
  device: "auto"  # Automatically distribute across GPUs
```

## Results Interpretation

### Game Performance
- **Positive trend in rewards**: Model is learning to play better
- **Increasing episode length**: Model is making more strategic moves
- **Win rate > 50%**: Model beats random opponent consistently

### Coding Performance
- **Stable or improving pass@1**: Model retains/improves coding ability
- **Large drops**: Catastrophic forgetting - reduce learning rate
- **No improvement**: May need more diverse prompts or different approach

## Citation

If you use this code, please cite:

```bibtex
@misc{qwen-connectx-rl,
  title={Qwen ConnectX Reinforcement Learning Pipeline},
  year={2024},
  url={https://github.com/your-repo/qwen-connectx-rl}
}
```