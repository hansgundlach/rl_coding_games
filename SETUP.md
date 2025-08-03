# Simple Setup Guide for H100

## Prerequisites
- H100 GPU with internet access
- Python 3.9+
- CUDA installed

## Setup Commands

```bash
# 1. Clone and enter directory
git clone <your-repo> game_project_rl && cd game_project_rl

# 2. Create virtual environment and install dependencies
python -m venv venv && source venv/bin/activate
pip install torch transformers datasets accelerate peft trl wandb PyYAML

# 3. Download MBPP dataset
mkdir -p evaluation/datasets/mbpp
wget -O evaluation/datasets/mbpp/mbpp.jsonl https://huggingface.co/datasets/mbpp/resolve/main/mbpp.jsonl

# 4. Download models (they'll cache automatically on first run)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.makedirs('./model_cache', exist_ok=True)
for model in ['Qwen/Qwen3-1.7B', 'Qwen/Qwen2.5-1.5B']:
    print(f'Downloading {model}...')
    AutoModelForCausalLM.from_pretrained(model, cache_dir='./model_cache')
    AutoTokenizer.from_pretrained(model, cache_dir='./model_cache')
print('Models cached!')
"

# 5. Set environment variables (optional)
export WANDB_API_KEY=your_key_here
export TOKENIZERS_PARALLELISM=false

# 6. Test setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 7. Run training
python grpo_code_game_icl.py --config configs/grpo_code_game_icl.yaml
```

## That's it!

The code will auto-detect H100 capabilities and use optimal settings. Models and datasets download automatically on first run.

## Quick config adjustments for better performance:

```bash
# Run with higher learning rate and more games
python grpo_code_game_icl.py --config configs/grpo_code_game_icl.yaml \
  --training_args.learning_rate=0.0001 \
  --training.games_per_step=16 \
  --training.num_steps=100
```