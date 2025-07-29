# Code Generation Game - Setup and Running Instructions

## Overview
This implements a code generation game where a Qwen model learns to write Python code that a static opponent (SmolLM-135M) cannot predict the output of. The system uses GRPO (Group Relative Policy Optimization) training.

## Quick Start

### 1. Prerequisites
- CUDA-compatible GPU (recommended)
- Python 3.10+
- Virtual environment activated

### 2. Environment Setup
```bash
# Navigate to project directory
cd /lambda/nfs/Gundlach2025/game_project_rl

# Activate virtual environment
source venv/bin/activate

# Setup environment variables (REQUIRED)
# Edit .env file and add your API keys:
# WANDB_API_KEY=your_wandb_key_here

# Verify packages are installed
python -c "import torch, transformers, trl, peft, wandb, dotenv; print('All packages available')"
```

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
1. **Trainable Model**: SmolLM-135M-Instruct with LoRA fine-tuning (3.5% trainable params)
2. **Static Opponent**: Same SmolLM-135M model (frozen) that tries to guess outputs
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

### Weights & Biases Dashboard
- **Project**: `qwen-code-game-grpo`
- **URL**: https://wandb.ai/hansgund-massachusetts-institute-of-technology/qwen-code-game-grpo
- **Metrics tracked**:
  - Training loss
  - Reward progression
  - Generated code examples
  - Model gradients

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