# CLAUDE.md - RL Training Project

This repository implements reinforcement learning training for code generation language models using GRPO (Group Relative Policy Optimization) and SPIRAL (Self-Play in Adversarial Language) algorithms.

## 🎯 Project Overview

**Primary Goal**: Train language models to generate code that is difficult for other models to predict, using adversarial self-play and reward mechanisms based on code execution success.

**Key Algorithms**:
- **GRPO**: Group Relative Policy Optimization for code execution rewards
- **SPIRAL**: Self-Play in Adversarial Language training with role-conditioned advantage estimation

## 📁 Repository Structure

### Core Training Scripts
- `grpo_code_execution.py` - Main GRPO training with code execution rewards
- `grpo_code_game.py` - GRPO adversarial game between models  
- `spiral_self_play.py` - SPIRAL self-play training implementation

### Configuration
- `configs/grpo_code_execution.yaml` - GRPO training configuration
- `configs/spiral_self_play.yaml` - SPIRAL training configuration

### Evaluation System (`evaluation/`)
- `mbpp/evaluator.py` - MBPP (Mostly Basic Python Problems) evaluation
- `configs/loader.py` - Configuration management with environment variable support
- `datasets/` - MBPP dataset management
- `tests/` - Comprehensive test suite

### Key Utilities
- `utils/env_loader.py` - Environment variable and API key management
- `utils/vllm_client.py` - vLLM integration for faster inference
- `agents/` - Model agent implementations

## 🚀 Quick Start

### Running GRPO Code Execution Training
```bash
python grpo_code_execution.py --config configs/grpo_code_execution.yaml
```

### Running SPIRAL Self-Play Training  
```bash
python spiral_self_play.py --config configs/spiral_self_play.yaml
```

### SLURM Submission (MIT Supercloud)
```bash
sbatch submit_grpo_training.sh
```

## 🔧 Platform Support

**Automatic Detection**: Scripts auto-detect platform and GPU capabilities

### MIT Supercloud V100
- Offline mode enabled automatically
- Conservative memory settings (smaller batch sizes)
- Local model caching in `./model_cache/`
- Reduced evaluation problem counts

### Lambda Labs A100/H100
- Online mode with W&B logging
- Aggressive memory utilization
- BF16 precision support
- Larger evaluation sets

## 📊 Evaluation & Monitoring

### MBPP Evaluation
- **Initial**: Before training starts
- **Interval**: Every N steps (configurable via `eval_interval_steps`)
- **Final**: After training completes

### W&B Integration
- Automatic logging of training metrics
- MBPP evaluation results: `mbpp_eval/pass_rate`, `mbpp_eval/problems_passed`
- Reward function metrics: `reward/avg_batch_reward`, `reward/successful_predictions_count`

### Recent Fix Applied
✅ **Fixed interval evaluations**: Added `IntervalEvaluationCallback` to `grpo_code_execution.py` to enable MBPP evaluations every 20 steps during training, properly logged to W&B for time-series analysis.

## 🎮 Training Approaches

### GRPO Code Execution (`grpo_code_execution.py`)
**Reward Function**: Based on code execution success
- `+1.0`: Code runs successfully with output
- `+0.5`: Code runs successfully without output  
- `-0.5`: Syntax/indentation errors
- `-1.0`: Runtime errors or timeouts

### GRPO Adversarial Game (`grpo_code_game.py`)
**Reward Function**: Adversarial prediction game
- `+1`: Opponent fails to predict code output correctly
- `-1`: Opponent successfully predicts code output

### SPIRAL Self-Play (`spiral_self_play.py`)
**Role-Conditioned Advantage Estimation**: Separate baselines for each player role to reduce variance in adversarial tic-tac-toe games.

## 🛠 Key Configuration Options

### Model Settings
```yaml
model:
  id: "Qwen/Qwen2.5-1.5B"  # Base model (auto-selects 1.5B -> 3B)
  cache_dir: "./model_cache"
```

### Training Parameters (Auto-adjusted by GPU type)
```yaml
training_args:
  learning_rate: 0.00002
  per_device_train_batch_size: 8  # Reduced on V100
  max_steps: 90
  num_generations: 2  # GRPO requirement
```

### MBPP Evaluation
```yaml
evaluation:
  enabled: true
  num_questions: 5  # Reduced on V100
  eval_interval_steps: 20  # Run every 20 steps
  temperature: 0.2
  timeout_seconds: 10
```

### vLLM Integration (Optional)
```yaml
vllm:
  enabled: false  # Set to true for faster inference
  gpu_memory_utilization: 0.85
  max_model_len: 2048
```

## 📈 Recent Training Results

Based on SLURM job logs, recent training runs show:
- **Initial MBPP Pass Rates**: 0.4 (2/5 problems)
- **Final MBPP Pass Rates**: Up to 1.0 (5/5 problems) 
- **Training Duration**: ~90-120 steps on V100 GPUs
- **Successful W&B Integration**: Metrics properly logged

## 🔍 Monitoring & Debugging

### Log Locations
- `logs/job_*/training_output.log` - Training progress
- `logs/job_*/mbpp_*.json` - Detailed evaluation results
- `slurm-grpo-*.out` - SLURM job outputs

### Key Debug Information
- Platform auto-detection results
- GPU memory utilization
- MBPP evaluation debug output
- W&B logging confirmation

## 🚨 Important Notes

### Memory Management
- V100 GPUs use aggressive memory reduction settings
- A100+ GPUs enable gradient checkpointing and larger batches
- Auto-detection prevents OOM errors across platforms

### Offline Mode (Supercloud)
- Automatically enabled on MIT Supercloud
- Uses cached models from `./model_cache/`
- W&B logging disabled in offline mode
- Local result storage in `./eval_results/`

### Code Safety
- Safe code execution with subprocess isolation
- Timeout protection (default 3-10 seconds)
- Memory limits and error capturing
- No malicious code execution capabilities

## 💡 Development Tips

### Adding New Training Scripts
1. Import evaluation system: `from evaluation import MBPPEvaluator, create_eval_config_for_training`
2. Use platform detection: `detect_platform_and_gpu()`
3. Add interval evaluation callbacks for metrics tracking
4. Follow existing patterns for W&B logging

### Environment Variables
```bash
export WANDB_API_KEY="your-key"  # For W&B logging
export MBPP_EVAL_NUM_QUESTIONS=10  # Override evaluation settings
export MBPP_DATASET_PATH="/custom/path"  # Custom dataset location
```

### Testing
```bash
python evaluation/tests/test_mbpp_system.py  # Test evaluation system
python tests/quick_test.py  # Quick pipeline test
```

This repository is actively maintained and optimized for both research and production RL training workflows on academic and commercial GPU clusters.