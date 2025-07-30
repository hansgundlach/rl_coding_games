# MBPP Evaluation System

This directory contains a comprehensive MBPP (Mostly Basic Python Problems) evaluation system for reinforcement learning training of code generation models.

## Directory Structure

```
evaluation/
├── README.md                   # This file
├── __init__.py                 # Main package exports
├── mbpp/                       # Core MBPP evaluation
│   ├── __init__.py
│   └── evaluator.py           # Main evaluator class
├── configs/                    # Configuration management
│   ├── __init__.py
│   ├── loader.py              # Configuration loading logic
│   └── eval_config.yaml       # Default configuration file
├── datasets/                   # Dataset management
│   ├── __init__.py            # Dataset utilities
│   └── sample_mbpp.json       # Sample dataset for testing
├── tests/                      # Test suite
│   ├── __init__.py
│   └── test_mbpp_system.py    # Comprehensive tests
└── results/                    # Evaluation results (auto-created)
```

## Quick Start

### Basic Usage

```python
from evaluation import MBPPEvaluator, create_eval_config_for_training

# Create configuration for your training script
config = create_eval_config_for_training("grpo_code_game")

# Initialize evaluator
evaluator = MBPPEvaluator(config)

# Run evaluation
if evaluator.should_evaluate(is_start=True):
    results = evaluator.evaluate_model(model, tokenizer, step=0, phase="initial")
    print(f"Pass rate: {results['pass_rate']:.3f}")
```

### Configuration

The system supports multiple configuration methods:

1. **YAML Configuration** (`evaluation/configs/eval_config.yaml`):
   ```yaml
   enabled: true
   num_questions: 10
   eval_at_start: true
   eval_at_end: true
   temperature: 0.2
   ```

2. **Environment Variables**:
   ```bash
   export MBPP_EVAL_NUM_QUESTIONS=20
   export MBPP_EVAL_TEMPERATURE=0.1
   export MBPP_DATASET_PATH="/path/to/mbpp.json"
   ```

3. **Code Overrides**:
   ```python
   config = create_eval_config_for_training(
       "grpo_code_game",
       num_questions=15,
       temperature=0.3
   )
   ```

### Platform-Specific Settings

The system automatically detects your platform and adjusts settings:

- **MIT Supercloud V100**: Conservative settings (5 questions, shorter timeouts)
- **Lambda A100**: Aggressive settings (20 questions, longer generations)
- **Other platforms**: Standard settings (10 questions, balanced)

## Dataset Setup

### Option 1: Automatic Download

```python
from evaluation.datasets import download_mbpp_dataset

# Download to default location
download_mbpp_dataset()

# Or specify custom directory
download_mbpp_dataset("/path/to/datasets/mbpp")
```

### Option 2: Manual Setup

1. Download the sanitized MBPP dataset:
   ```bash
   wget https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json
   ```

2. Place it in one of these locations:
   - `./evaluation/datasets/sanitized-mbpp.json`
   - `./data/mbpp/sanitized-mbpp.json`
   - Or specify path in config

## Evaluation Features

### Configurable Evaluation Schedule

- **Start of training**: `eval_at_start: true`
- **End of training**: `eval_at_end: true`  
- **Interval-based**: `eval_interval_steps: 100` (every 100 steps)
- **Disable entirely**: `enabled: false`

### Safe Code Execution

- Subprocess isolation
- Timeout protection (default 10s)
- Memory limits
- Error capturing

### Comprehensive Results

```json
{
  "step": 0,
  "phase": "initial",
  "total_problems": 10,
  "problems_passed": 7,
  "pass_rate": 0.7,
  "eval_time_seconds": 45.2,
  "config": {
    "num_questions": 10,
    "temperature": 0.2
  }
}
```

## Integration with Training Scripts

The evaluation system is already integrated into:

- `grpo_code_game.py`
- `grpo_code_execution.py`

### Key Integration Points

1. **Initialization**: Loads platform-specific config
2. **Start evaluation**: Before training begins
3. **End evaluation**: After training completes
4. **WandB logging**: Automatic metric logging
5. **Result storage**: Detailed results saved locally

## Testing

Run the comprehensive test suite:

```bash
python evaluation/tests/test_mbpp_system.py
```

Or run specific tests:

```bash
python -m unittest evaluation.tests.test_mbpp_system.TestMBPPEvaluator
```

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `MBPP_EVAL_ENABLED` | Enable/disable evaluation | `true` |
| `MBPP_EVAL_NUM_QUESTIONS` | Number of questions to evaluate | `10` |
| `MBPP_EVAL_AT_START` | Evaluate at training start | `true` |
| `MBPP_EVAL_AT_END` | Evaluate at training end | `true` |
| `MBPP_EVAL_INTERVAL_STEPS` | Steps between evaluations | `null` |
| `MBPP_DATASET_PATH` | Path to MBPP dataset | auto-detect |
| `MBPP_TEMPERATURE` | Generation temperature | `0.2` |
| `MBPP_MAX_NEW_TOKENS` | Maximum tokens to generate | `512` |
| `MBPP_TIMEOUT_SECONDS` | Code execution timeout | `10` |
| `MBPP_RESULTS_DIR` | Results directory | `./eval_results` |

## Troubleshooting

### Dataset Not Found

```
⚠️ MBPP evaluation disabled - dataset not found
```

**Solution**: Download the dataset using one of the methods above.

### Import Errors

```python
from evaluation import MBPPEvaluator  # ✅ Correct
from evaluation.mbpp_evaluator import MBPPEvaluator  # ❌ Old path
```

### Memory Issues on V100

The system automatically reduces batch sizes and question counts on V100 GPUs. For manual override:

```python
config = create_eval_config_for_training(
    "grpo_code_game", 
    num_questions=3,  # Even smaller
    timeout_seconds=5
)
```

## Contributing

When adding new features:

1. Add tests to `evaluation/tests/test_mbpp_system.py`
2. Update configuration options in `evaluation/configs/eval_config.yaml`
3. Document changes in this README
4. Ensure compatibility with both V100 and A100 environments