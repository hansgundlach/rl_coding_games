# Qwen ConnectX RL Training 🎮🤖

**Train a Qwen language model to play Connect-X using PPO reinforcement learning with LoRA fine-tuning**

This project implements legitimate RL training where a **Qwen 2.5-3B model** learns to play Connect-X through reinforcement learning, while being evaluated on coding benchmarks to measure knowledge retention.

## ✨ Features

- **🧠 Real Qwen Model Integration**: Uses Qwen 2.5-3B with proper text-to-action conversion
- **🎯 LoRA Fine-tuning**: Efficient parameter updates (~30M trainable parameters)
- **🔄 PPO Training**: Proximal Policy Optimization for stable RL learning
- **🎮 Connect-X Gameplay**: Full game environment with reward shaping
- **📊 Coding Evaluation**: HumanEval-Plus and MBPP-Plus benchmark integration
- **📈 W&B Logging**: Complete training metrics and coding performance tracking
- **🛡️ Error Handling**: Robust resource management and fallback systems

## 🚀 Quick Start

### 1. Automated Setup (Recommended)
```bash
# Run automated installation
./install.sh

# Or check your setup
python check_setup.py
```

### 2. Manual Setup
```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Login to Hugging Face (required for Qwen model)
huggingface-cli login

# Login to W&B (optional, for logging)
wandb login
```

### 3. Start Training
```bash
# Quick test (2 epochs, 5 episodes each)
python training/qwen_ppo_train.py --config configs/qwen_ppo.yaml

# Expected output:
# Loading Qwen model: Qwen/Qwen2.5-3B
# trainable params: 29,933,568 || all params: 3,115,872,256 || trainable%: 0.96
# Starting Qwen PPO training...
# Epoch 1/2
#   Episode 5: Reward=-82.00, Length=11
#   Mean reward: -12.60 ± 96.57
#   Policy loss: 3.5439
#   Saved LoRA checkpoint: checkpoints/qwen_ppo/qwen_ppo_epoch_1
```

## 📊 What Makes This Legitimate RL Training

### ✅ Real Implementation vs ❌ Mock
| Component | This Project | Previous Mock |
|-----------|-------------|---------------|
| **Qwen Model** | ✅ Actual Qwen 2.5-3B loading & inference | ❌ Simple heuristics |
| **LoRA Training** | ✅ Real PEFT integration (30M params) | ❌ Mock objects |
| **Game → Text** | ✅ Board state to natural language prompts | ❌ Direct numeric input |
| **Text → Action** | ✅ Parse model output to valid moves | ❌ Random selection |
| **Coding Eval** | ✅ Real EvalPlus benchmarks | ❌ Static fake scores |
| **PPO Integration** | ✅ Reward-weighted loss on LLM | ❌ Standard RL networks |

### 🎯 Training Process
1. **Game State → Prompt**: Convert Connect-X board to natural language
2. **Qwen Inference**: Generate text response for move selection  
3. **Action Parsing**: Extract valid column number from generated text
4. **Reward Collection**: Play full episodes and collect rewards
5. **LoRA Updates**: Update model parameters based on game performance
6. **Coding Evaluation**: Periodically test HumanEval-Plus/MBPP-Plus performance

## 📁 Project Structure

```
game_project_rl/
├── 🚀 Quick Start
│   ├── install.sh              # Automated setup script
│   ├── check_setup.py          # Verify installation
│   └── test_pipeline.py        # End-to-end testing
├── 🧠 Core Training
│   ├── training/qwen_ppo_train.py    # Main Qwen PPO trainer
│   ├── agents/qwen_policy.py         # Real Qwen agent implementation  
│   ├── training/lora_utils.py        # LoRA integration utilities
│   └── configs/qwen_ppo.yaml         # Training configuration
├── 🎮 Environment
│   ├── environment/connectx_wrapper.py  # Connect-X game environment
│   └── environment/reward_shaping.py    # Reward engineering
├── 📊 Evaluation  
│   ├── evaluation/evalplus_runner.py    # Real coding benchmarks
│   └── training/wandb_logger.py         # W&B metrics logging
├── 📚 Documentation
│   ├── COMPLETE_SETUP_GUIDE.md      # Detailed setup instructions
│   ├── USAGE_GUIDE.md               # Usage and configuration
│   └── README.md                    # This file
└── 💾 Generated
    ├── checkpoints/qwen_ppo/        # LoRA checkpoints (~50MB each)
    ├── model_cache/                 # Cached Qwen model (~6GB)
    └── wandb/                       # Training logs
```

## ⚙️ Configuration

### Resource Requirements
| Model | LoRA Rank | Batch Size | GPU Memory | Speed |
|-------|-----------|------------|------------|-------|
| **Qwen2.5-1.5B** | 8 | 2 | ~6GB | Fast |
| **Qwen2.5-3B** | 16 | 2 | ~12GB | Medium |
| **Qwen2.5-3B** | 16 | 8 | ~16GB | Medium |

### Key Settings (`configs/qwen_ppo.yaml`)
```yaml
# Model configuration
qwen:
  model_name: "Qwen/Qwen2.5-3B"  # Or "Qwen/Qwen2.5-1.5B" for less memory
  device: "auto"                  # Auto-detect GPU/CPU
  temperature: 0.7                # Generation randomness

# LoRA settings
lora:
  r: 16                          # Rank (8/16/32/64)
  lora_alpha: 32                 # Scaling (usually 2x rank)
  target_modules: ["q_proj", "k_proj", "v_proj", ...]

# Training settings  
training:
  num_epochs: 50                 # Total epochs
  episodes_per_epoch: 20         # Games per epoch
  batch_size: 8                  # Adjust for GPU memory
  learning_rate: 5e-5            # LoRA learning rate
```

## 📈 Results & Monitoring

### W&B Dashboard
Training automatically logs to Weights & Biases:
- **Game Performance**: Reward progression, episode length, win rate
- **Training Metrics**: Policy loss, gradient norms, learning curves  
- **Coding Evaluation**: HumanEval-Plus and MBPP-Plus scores over time
- **System Metrics**: GPU utilization, memory usage

### Expected Training Progression
```
Epoch 1:  Mean reward: -12.60 ± 96.57, Policy loss: 3.54  # Learning phase
Epoch 5:  Mean reward: 25.40 ± 45.32, Policy loss: 2.12   # Improvement
Epoch 20: Mean reward: 78.90 ± 23.15, Policy loss: 1.05   # Competent play
```

## 🔧 Advanced Usage

### Production Training
```bash
# Edit configs/qwen_ppo.yaml for longer training:
# training.num_epochs: 50
# training.episodes_per_epoch: 20  
# evaluation.skip_initial_eval: false

python training/qwen_ppo_train.py --config configs/qwen_ppo.yaml
```

### Evaluation Only  
```python
from evaluation.evalplus_runner import EvalPlusRunner

runner = EvalPlusRunner(
    model_path="Qwen/Qwen2.5-3B",
    lora_path="checkpoints/qwen_ppo/qwen_ppo_epoch_10"
)
results = runner.run_evaluation()
print(results)  # {'humaneval_plus': {'pass@1': 0.25, ...}, ...}
```

### Custom Reward Shaping
```python
# Edit environment/reward_shaping.py
def compute_reward(self, board, action, done, winner):
    reward = 0
    if done:
        reward = 100 if winner == self.player else -100
    # Add your custom intermediate rewards here
    return reward
```

## 🐛 Troubleshooting

### Common Issues
```bash
# GPU out of memory
# → Reduce batch_size in config (8→4→2→1)
# → Use smaller model: "Qwen/Qwen2.5-1.5B"

# Model download fails  
# → Check: huggingface-cli whoami
# → Login: huggingface-cli login

# Training doesn't start
# → Verify setup: python check_setup.py
# → Check logs in wandb/ directory
```

### Performance Tips
- **Memory**: Reduce `batch_size` and `episodes_per_epoch` for limited GPU
- **Speed**: Set `evaluation.skip_initial_eval: true` for faster startup
- **Quality**: Increase `lora.r` and `training.num_epochs` for better performance

## 📚 Documentation

- **[COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md)** - Detailed installation with troubleshooting
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Configuration options and advanced usage
- **[Weights & Biases Dashboard](https://wandb.ai)** - Live training metrics

## 🎯 Key Achievements

This project solves the issues from the original mock implementation:

✅ **Real Qwen Integration** - Actual model loading, inference, and LoRA training  
✅ **Legitimate RL Training** - PPO updates based on Connect-X game rewards  
✅ **Text-based Gameplay** - Natural language prompts and response parsing  
✅ **Coding Evaluation** - Real HumanEval-Plus and MBPP-Plus benchmarks  
✅ **Resource Management** - Proper error handling and GPU memory optimization  
✅ **Complete Pipeline** - End-to-end training with checkpointing and logging  

## 🤝 Contributing

1. Run tests: `python test_pipeline.py`  
2. Check setup: `python check_setup.py`
3. Follow configuration patterns in `configs/`
4. Add new agents in `agents/` directory
5. Extend evaluation in `evaluation/`

---

**Ready to train a language model to play games? 🎮**

```bash
python training/qwen_ppo_train.py --config configs/qwen_ppo.yaml
```