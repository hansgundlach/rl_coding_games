# Complete Setup Guide - Qwen ConnectX RL Training

## 🎯 Overview

This project trains a **Qwen 2.5-3B language model** to play Connect-X using **PPO reinforcement learning** with **LoRA fine-tuning**. The model is evaluated on both game performance and coding benchmarks (HumanEval-Plus, MBPP-Plus).

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **Python**: 3.10 or 3.11 (3.12 may have compatibility issues)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- **RAM**: 16GB+ system RAM (32GB+ recommended)
- **Storage**: 20GB+ free space (for models and checkpoints)

### Required Accounts
- **Hugging Face**: For downloading Qwen models
- **Weights & Biases**: For training logs and metrics (optional but recommended)
- **GitHub**: For code access (if cloning from repository)

## 🚀 Step-by-Step Installation

### 1. Environment Setup

```bash
# Navigate to project directory
cd /home/ubuntu/game_project_rl

# Create Python virtual environment
python3.10 -m venv venv
# OR for conda users:
# conda create -n qwen-rl python=3.10

# Activate environment
source venv/bin/activate
# OR for conda: conda activate qwen-rl

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2. Core Dependencies Installation

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all project dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### 3. Hugging Face Setup

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login to Hugging Face (required for Qwen model access)
huggingface-cli login
# Enter your HF token when prompted

# Alternative: Set environment variable
export HUGGINGFACE_HUB_TOKEN="your_hf_token_here"
```

**Getting your Hugging Face Token:**
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token for use above

### 4. Weights & Biases Setup (Optional but Recommended)

```bash
# Install W&B
pip install wandb

# Login to W&B
wandb login
# Enter your W&B API key when prompted

# Alternative: Set environment variable
export WANDB_API_KEY="your_wandb_api_key_here"
```

**Getting your W&B API Key:**
1. Go to [wandb.ai](https://wandb.ai) and create account
2. Go to [wandb.ai/authorize](https://wandb.ai/authorize)
3. Copy your API key

### 5. Model Download and Verification

The Qwen model (~6GB) will be downloaded automatically on first run, but you can pre-download it:

```bash
# Pre-download Qwen model (optional)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Qwen model...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', cache_dir='./model_cache')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B', cache_dir='./model_cache')
print('Download complete!')
"
```

### 6. Installation Verification

```bash
# Run comprehensive test suite
python test_pipeline.py

# Expected output:
# === Qwen ConnectX RL Pipeline Test ===
# 
# Testing imports...
# ✓ PyTorch 2.x.x
# ✓ Transformers 4.x.x
# ✓ PEFT 0.x.x
# ✓ QwenPolicyAgent
# ✓ LoRA utilities
# ✓ EvalPlus runner
# ✓ ConnectX environment
# 
# Checking GPU availability...
# ✓ CUDA available with 1 GPU(s)
#   GPU 0: NVIDIA ... (XGB)
# 
# Testing configuration files...
# ✓ PPO config loaded
# ✓ Qwen PPO config loaded
# ✓ Configuration validation passed
# 
# Testing ConnectX environment...
# ✓ Environment reset - state shape: (6, 7)
# ✓ Environment step - reward: 0.0, done: False
# 
# Testing LoRA setup...
# ✓ LoRA utils working - trainable params: 11
# 
# Testing Qwen agent...
# Loading Qwen model: Qwen/Qwen2.5-3B
# Available GPU memory: X.X GB
# Qwen model loaded successfully
# ✓ Agent generated action: X
# 
# === Test Summary ===
# Imports: PASS
# GPU Check: PASS
# Configuration: PASS
# Environment: PASS
# LoRA Setup: PASS
# Qwen Agent: PASS
# 
# Overall: 6/6 tests passed
# 🎉 All tests passed! The pipeline is ready to use.
```

## 🎮 Quick Start - Running Training

### 1. Basic Training Run (Recommended for first try)

```bash
# Start training with reduced configuration for testing
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

### 2. Production Training Configuration

For longer training, edit `configs/qwen_ppo.yaml`:

```yaml
# Training settings for production
training:
  num_epochs: 50        # Increase from 2
  episodes_per_epoch: 20  # Increase from 5
  batch_size: 8         # Increase from 2
  learning_rate: 5e-5
  weight_decay: 0.01
  max_grad_norm: 1.0

# Checkpointing
checkpoints:
  save_dir: "checkpoints/qwen_ppo"
  save_interval: 5      # Save every 5 epochs instead of 1

# Evaluation
evaluation:
  skip_initial_eval: false  # Enable coding evaluation
  coding_eval_interval: 10  # Evaluate every 10 epochs
  num_coding_samples: 10    # More samples for better evaluation
```

## 📁 What Gets Downloaded/Created

### Automatic Downloads (First Run)
```
model_cache/                    # ~6GB - Qwen model files
├── models--Qwen--Qwen2.5-3B/
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
```

### Generated During Training
```
checkpoints/qwen_ppo/          # LoRA checkpoints (~50MB each)
├── qwen_ppo_epoch_5/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── README.md
├── qwen_ppo_epoch_10/
└── ...

wandb/                         # Training logs
├── run-*/
└── ...
```

## 🔧 Configuration Guide

### Key Configuration Files

1. **`configs/qwen_ppo.yaml`** - Main training configuration
2. **`configs/ppo.yaml`** - Original PPO configuration (comparison)

### Important Settings to Adjust

```yaml
# Model settings
qwen:
  model_name: "Qwen/Qwen2.5-3B"  # Could use "Qwen/Qwen2.5-1.5B" for less memory
  device: "auto"                  # "cuda", "cpu", or "auto"
  temperature: 0.7                # Lower = more deterministic

# LoRA settings - affect model capacity and training speed
lora:
  r: 16          # Higher = more parameters (8, 16, 32, 64)
  lora_alpha: 32 # Scaling factor (usually 2x r)
  lora_dropout: 0.1

# Training settings - affect training time and performance
training:
  num_epochs: 50           # Total training epochs
  episodes_per_epoch: 20   # Games per epoch
  batch_size: 8           # Adjust based on GPU memory
  learning_rate: 5e-5     # LoRA learning rate

# Memory optimization
checkpoints:
  save_interval: 5  # Save less frequently to save disk space
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```bash
# Error: RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `batch_size` in config (try 4, 2, or 1)
- Use smaller model: `"Qwen/Qwen2.5-1.5B"`
- Set `qwen.device: "cpu"` (much slower)

#### 2. Model Download Fails
```bash
# Error: HTTPError: 401 Client Error: Unauthorized
```
**Solutions:**
- Check Hugging Face login: `huggingface-cli whoami`  
- Re-login: `huggingface-cli login`
- Check internet connection

#### 3. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'transformers'
```
**Solutions:**
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`
- Install in development mode: `pip install -e .`

#### 4. W&B Authentication
```bash
# Error: wandb.errors.AuthenticationError
```
**Solutions:**
- Login to W&B: `wandb login`
- Set offline mode: `export WANDB_MODE=offline`
- Disable W&B: Comment out W&B logger in code

#### 5. Slow Training
**Solutions:**
- Reduce `episodes_per_epoch` and `num_epochs` for testing
- Set `evaluation.skip_initial_eval: true`
- Use GPU instead of CPU
- Reduce `coding_eval_interval`

### Memory Requirements by Configuration

| Model Size | LoRA Rank | Batch Size | GPU Memory | Training Speed |
|------------|-----------|------------|------------|----------------|
| Qwen2.5-1.5B | 8 | 2 | ~6GB | Fast |
| Qwen2.5-1.5B | 16 | 4 | ~8GB | Fast |
| Qwen2.5-3B | 16 | 2 | ~12GB | Medium |
| Qwen2.5-3B | 16 | 8 | ~16GB | Medium |
| Qwen2.5-3B | 32 | 8 | ~20GB | Slow |

## 📊 Monitoring Training

### W&B Dashboard
Visit your W&B project URL (shown in training output):
- **Game Performance**: Reward progression, episode length
- **Training Metrics**: Policy loss, gradient norms
- **Coding Evaluation**: HumanEval-Plus and MBPP-Plus scores
- **System Metrics**: GPU utilization, memory usage

### Local Monitoring
```bash
# Check GPU usage
nvidia-smi

# Monitor training logs
tail -f wandb/latest-run/logs/debug.log

# Check checkpoint sizes
du -h checkpoints/qwen_ppo/
```

## 🎯 Expected Results

### Training Progression
- **Initial episodes**: Negative rewards (random play)
- **After 5-10 epochs**: Occasional positive rewards
- **After 20+ epochs**: Consistently positive rewards
- **Policy loss**: Should decrease over time (3.0 → 1.0 → 0.5)

### Coding Performance
- **Baseline Qwen**: ~15-25% pass@1 on HumanEval-Plus
- **After RL training**: May maintain or slightly change
- **Goal**: Maintain coding ability while learning game strategy

## 🚀 Advanced Usage

### Custom Model Training
```bash
# Use different model size
# Edit configs/qwen_ppo.yaml:
# qwen.model_name: "Qwen/Qwen2.5-1.5B"  # Smaller, faster
# qwen.model_name: "Qwen/Qwen2.5-7B"    # Larger, better (needs more GPU)
```

### Evaluation Only
```bash
# Evaluate existing checkpoint without training
python -c "
from evaluation.evalplus_runner import EvalPlusRunner
runner = EvalPlusRunner(
    model_path='Qwen/Qwen2.5-3B',
    lora_path='checkpoints/qwen_ppo/qwen_ppo_epoch_10'
)
results = runner.run_evaluation()
print(results)
"
```

### Batch Training with Different Configs
```bash
# Create multiple config files and run in sequence
for config in configs/qwen_ppo_*.yaml; do
    echo "Training with $config"
    python training/qwen_ppo_train.py --config "$config"
done
```

## 📞 Support

If you encounter issues:

1. **Check this guide** for common solutions
2. **Run the test script**: `python test_pipeline.py`
3. **Check system requirements** match your setup
4. **Verify all dependencies** are installed correctly
5. **Monitor GPU memory** usage during training

The setup is complete when `test_pipeline.py` shows all tests passing! 🎉