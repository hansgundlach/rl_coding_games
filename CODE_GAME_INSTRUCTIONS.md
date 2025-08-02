# Code Generation Game - Setup and Running Instructions

## Overview

## Quick Start

### 1. Prerequisites

- CUDA-compatible GPU (recommended)
- Python 3.10+
- Virtual environment activated

### 2. Platform-Specific Setup

#### MIT Supercloud (V100, No Internet) - RECOMMENDED SETUP

**Step 1: Request Interactive Session**

```bash
# Request GPU node with adequate resources for training
LLsub -i -s 4 -g volta:1 -R "rusage[mem=64000]" -W 4:00

# Wait for node assignment, then SSH to the assigned node
ssh your-assigned-node-name
```

**Step 2: Set Up Environment**

```bash
# Navigate to your project directory
cd /home/gridsan/yourusername/game_project_rl

# Activate virtual environment
source venv/bin/activate

# Verify Qwen2.5-1.5B model is cached (should be available now)
ls -la model_cache/models--Qwen--Qwen2.5-1.5B/

# Check GPU allocation
nvidia-smi
```

**Step 3: Run Training**

```bash
# Just run - auto-detects Supercloud and uses cached models!
python grpo_code_game.py
```

**Expected Supercloud Output:**

```
🔍 Auto-detected: Supercloud platform
🎮 GPU: V100, BF16 support: False
🌐 Offline mode: True (detected Supercloud environment)
🎯 Using preferred cached model: Qwen/Qwen2.5-1.5B (better memory efficiency)
Loading trainable model: Qwen/Qwen2.5-1.5B
🔧 Using V100 + 1.5B model settings (moderate memory reduction)
🚫 Skipping W&B login (offline mode)
Starting GRPO training...
```

**Supercloud Resource Requirements:**

- **GPU**: 1x Tesla V100 (32GB VRAM)
- **CPU**: 4 cores recommended
- **Memory**: 64GB RAM recommended
- **Time**: 4+ hours for full training

**Memory Settings (V100 + Qwen2.5-1.5B):**

- `batch_size=4` (moderate memory usage)
- `gradient_accumulation_steps=2`
- `max_prompt_length=256`
- `max_completion_length=128`
- `gradient_checkpointing=True` (memory optimization)
- `fp16=True` (V100 compatible precision)

#### Lambda Labs (A100, Internet Access)

```bash
# Navigate to project directory
cd /lambda/nfs/Gundlach2025/game_project_rl

# Activate virtual environment
source venv/bin/activate

# Run training - auto-detects Lambda environment
python grpo_code_game.py
```

**Expected Lambda Output:**

```
🔍 Auto-detected: Lambda platform
🎮 GPU: A100, BF16 support: True
🌐 Offline mode: False (internet connection available)
✓ Logged into W&B using environment variable
Loading trainable model: Qwen/Qwen2.5-1.5B
🔧 Using standard memory settings for A100/other GPUs
Starting GRPO training...
```

#### Automatic Platform Detection

The system **automatically detects** your platform and configures itself:

- **🔍 Platform Detection**: Detects Supercloud vs Lambda based on hostname/path
- **🎮 GPU Detection**: Auto-detects V100, A100, H100 and configures BF16 accordingly  
- **🌐 Network Detection**: Tests internet connectivity and enables offline mode as needed
- **🎯 Model Selection**: Prefers Qwen2.5-1.5B over 3B for better memory efficiency
- **⚙️ Memory Optimization**: Adjusts batch size and settings based on GPU + model combination

**No environment variables needed - just run `python grpo_code_game.py`!**

### 2.1. Configure API Keys

Create or edit the `.env` file in the project root:

```bash
# .env
WANDB_API_KEY=your_wandb_api_key_here

# Optional: Other API keys  
# OPENAI_API_KEY=your_openai_key_here
# HF_TOKEN=your_huggingface_token_here
```

**Important**: Never commit the `.env` file to version control!

### 3. Run the Code Generation Game

```bash
# Simple run (uses default settings)
python grpo_code_game.py
```

## How It Works

### Game Mechanics

3. **Task**: Generate Python code that returns an integer, formatted like:

   ```
   ```python
   def tricky():
       return int('0b1011', 2)
   print(tricky())
   ```

   ```output
   11
   ```

   ```

### Reward System

- **+1 reward**: Static opponent guesses the output incorrectly
- **-1 reward**: Static opponent guesses the output correctly
- **Goal**: Learn to write increasingly tricky code that fools the opponent

### Training Process

1. Model generates code completion
2. Code is executed to get true output
3. Static opponent tries to predict the output
4. Reward calculated based on whether opponent was fooled
5. GRPO updates the model to maximize rewards

## Expected Training Output

### Initial Setup

```
Created dataset: Dataset({
    features: ['prompt'],
    num_rows: 1000
})
Loading trainable model: HuggingFaceTB/SmolLM-135M-Instruct
trainable params: 4,884,480 || all params: 139,399,488 || trainable%: 3.5039
Loading static opponent: HuggingFaceTB/SmolLM-135M-Instruct
Setting up GRPO training...
Starting GRPO training...
```

### During Training

You'll see debug output for the first few examples:

```
==================================================
Model prediction: 42
Code: def simple(): return 42
print(simple())
Expected output: 42
True output: 42
Original completion: ```python
def simple():
    return 42
print(simple())
```

```output
42
```...
==================================================
```

### Training Progress

- **Training steps**: 500 total (1000 samples / batch_size=8 * num_generations=8)
- **Time per step**: ~45-50 seconds initially
- **Total training time**: ~6-7 hours for full run

## Monitoring and Logging

### Weights & Biases (W&B) for Offline Use

By default, the training script runs in offline mode to prevent network issues on the Supercloud. The logs are saved locally in the `wandb/` directory.

To sync your results to the W&B dashboard later, you can use the `wandb sync` command from your local machine (with internet access):

```bash
# On your local machine, after downloading the wandb directory
wandb sync /path/to/your/project/wandb/offline-run-*
```

To disable W&B logging entirely, you can set the `WANDB_DISABLED` environment variable:

```bash
export WANDB_DISABLED=true
python grpo_code_game.py
```

### Local Checkpoints

Saved to: `checkpoints/grpo_code/`

- Periodic model checkpoints
- Final trained model

## Configuration Options

### Model Settings

```python
# In grpo_code_game.py, modify these variables:
model_id = "HuggingFaceTB/SmolLM-135M-Instruct"  # Change model size
lora_config = LoraConfig(
    r=16,           # LoRA rank (8/16/32)
    lora_alpha=32,  # LoRA scaling
)
```

### Training Settings

```python
training_args = GRPOConfig(
    output_dir="checkpoints/grpo_code",
    learning_rate=2e-5,                    # Learning rate
    per_device_train_batch_size=8,         # Batch size
    gradient_accumulation_steps=2,         # Gradient accumulation
    max_prompt_length=512,                 # Input length
    max_completion_length=200,             # Output length
    num_generations=8,                     # Generations per batch
    num_train_epochs=1,                    # Epochs
)
```

### Dataset Settings

```python
# Number of training prompts
dataset = Dataset.from_dict({
    "prompt": [prompt] * 1000  # Change 1000 to desired size
})
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```bash
   # Reduce batch size
   per_device_train_batch_size=4  # or 2, 1
   ```

2. **Model Loading Errors**

   ```bash
   # Check model cache
   ls model_cache/
   # Clear cache if needed
   rm -rf model_cache/
   ```

3. **W&B Authentication**

   ```bash
   # Check login status
   wandb status
   # Re-login if needed
   wandb login --relogin
   ```

4. **Import Errors**

   ```bash
   # Verify environment
   source venv/bin/activate
   pip list | grep -E "(torch|transformers|trl|peft)"
   ```

### Performance Tips

1. **Speed up training**:
   - Use smaller dataset (500 instead of 1000)
   - Reduce `num_generations` to 4
   - Increase `gradient_accumulation_steps`

2. **Improve results**:
   - Increase LoRA rank (`r=32`)
   - Run multiple epochs (`num_train_epochs=3`)
   - Use larger base model

3. **Monitor progress**:
   - Check W&B dashboard regularly
   - Look for increasing reward trends
   - Examine generated code quality

## Expected Results

### Training Progression

- **Early**: Simple code, often guessed correctly (negative rewards)
- **Mid-training**: More complex patterns emerge
- **Late**: Sophisticated tricks that fool the opponent (positive rewards)

### Sample Evolved Code

The model should learn to generate increasingly sophisticated code:

```python
# Early training - simple
def f(): return 42

# Mid training - basic tricks  
def f(): return len("hello")

# Late training - complex
def f(): return sum(ord(c) for c in "AI") % 100
```

## Next Steps

1. **Experiment with different models**:
   - Try larger models (Qwen2.5-1.5B, Qwen2.5-3B)
   - Different opponent models

2. **Modify the game**:
   - Change reward structure
   - Add complexity penalties
   - Multi-round games

3. **Evaluate results**:
   - Test against different opponents
   - Measure code complexity
   - Human evaluation of creativity

## File Structure

```
game_project_rl/
├── grpo_code_game.py           # Main training script
├── CODE_GAME_INSTRUCTIONS.md   # This file
├── environment/
│   └── code_game_wrapper.py    # Game environment
├── agents/
│   └── qwen_code_agent.py      # Agent implementations
├── checkpoints/
│   └── grpo_code/              # Model checkpoints
└── wandb/                      # Training logs
```

## Support

For issues:

1. Check the troubleshooting section above
2. Examine W&B logs for training insights
3. Verify GPU memory usage with `nvidia-smi`
4. Check model outputs in the debug prints
