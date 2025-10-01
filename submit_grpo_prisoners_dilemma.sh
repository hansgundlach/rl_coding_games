#!/bin/bash
#SBATCH --job-name=grpo_prisoners_dilemma
#SBATCH --output=logs/job_%j_%x/slurm-grpo-prisoners-dilemma-%j.out
#SBATCH --error=logs/job_%j_%x/slurm-grpo-prisoners-dilemma-%j.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# MIT Supercloud SLURM script for GRPO Prisoner's Dilemma Training
# Single V100 GPU training with self-play against frozen opponent
#
# Usage: sbatch submit_grpo_prisoners_dilemma.sh [config_overrides...]
# Examples:
#   sbatch submit_grpo_prisoners_dilemma.sh --training.num_steps=50 --training.opponent_refresh_steps=5
#   sbatch submit_grpo_prisoners_dilemma.sh --evaluation.num_questions=10 --game.wsls_bot_prob=0.3

# Parse command line arguments for config overrides
CONFIG_OVERRIDES=""
echo "🔍 Debug: Parsing arguments: $@"
for arg in "$@"; do
    echo "🔍 Debug: Processing argument: '$arg'"
    if [[ $arg == --* ]]; then
        # Keep the -- prefix for Python script
        CONFIG_OVERRIDES="$CONFIG_OVERRIDES $arg"
        echo "🔍 Debug: Added override: $arg"
    fi
done
echo "🔍 Debug: Final CONFIG_OVERRIDES: '$CONFIG_OVERRIDES'"

echo "🚀 Starting GRPO Prisoner's Dilemma Training on Supercloud V100"
echo "================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
if [ ! -z "$CONFIG_OVERRIDES" ]; then
    echo "Config overrides: $CONFIG_OVERRIDES"
fi
echo ""

# Environment setup
echo "🔧 Setting up environment..."
cd $SLURM_SUBMIT_DIR || cd /home/gridsan/hgundlach/game_project_rl
echo "📂 Working directory: $(pwd)"
echo "📂 Contents: $(ls -la | head -5)"

# Try to load modules if they exist (optional for Supercloud)
module purge 2>/dev/null || echo "⚠️ module command not available"
module load cuda/12.1 2>/dev/null || echo "⚠️ cuda/12.1 module not found, using system CUDA"
module load python/3.9 2>/dev/null || echo "⚠️ python/3.9 module not found, using system Python"

# Check if virtual environment is already active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "🐍 Virtual environment already active: $VIRTUAL_ENV"
else
    echo "🐍 Attempting to activate virtual environment..."
    # Try common venv locations
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✅ Activated venv/bin/activate"
    elif [ -f "../venv/bin/activate" ]; then
        source ../venv/bin/activate
        echo "✅ Activated ../venv/bin/activate"
    elif [ -f "$HOME/venv/bin/activate" ]; then
        source $HOME/venv/bin/activate
        echo "✅ Activated $HOME/venv/bin/activate"
    else
        echo "⚠️ No virtual environment found, using system Python"
        echo "   Checked: venv/bin/activate, ../venv/bin/activate, $HOME/venv/bin/activate"
        echo "   Current Python: $(which python)"
        echo "   If this fails, activate your venv before running sbatch"
    fi
fi

# Set offline environment variables for Supercloud
echo "🌐 Setting offline mode for Supercloud..."
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

# Add W&B project override capability
if [ -z "$WANDB_PROJECT" ]; then
    # Set consistent project name for grpo_prisoners_dilemma
    export WANDB_PROJECT="grpo-prisoners-dilemma"
    echo "📊 Set WANDB_PROJECT to: $WANDB_PROJECT"
else
    echo "📊 Using WANDB_PROJECT override: $WANDB_PROJECT"
fi

# Set up logs directory with job-specific subfolder
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_LOG_DIR="logs/grpo_prisoners_dilemma_job_${SLURM_JOB_ID}_${TIMESTAMP}"
mkdir -p "$JOB_LOG_DIR"
echo "📁 Created job log directory: $JOB_LOG_DIR"

# Export log directory for Python script to use
export GRPO_LOG_DIR="$JOB_LOG_DIR"

# GPU information
echo "🎮 GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️ nvidia-smi not found, checking CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
echo ""

# Memory monitoring before training
echo "💾 Pre-training Memory Status:"
echo "System Memory:"
free -h
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Memory:"
    nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "GPU Memory: nvidia-smi not available"
fi
echo ""

# Check if we have enough memory for large evaluations
if echo "$CONFIG_OVERRIDES" | grep -q "evaluation.num_questions=200"; then
    echo "⚠️  WARNING: Large evaluation (200 questions) detected!"
    echo "💡 Consider using quantization to reduce memory usage:"
    echo "   --model.quantization.enabled=true --model.quantization.load_in_4bit=true"
    echo ""
fi

# Python and package versions
echo "📦 Environment Information:"
python --version
pip list | grep -E "(torch|transformers|trl|peft|datasets)"
echo ""

# Start training with timestamping
echo "🏋️ Starting GRPO Prisoner's Dilemma training at $(date)..."
PYTHON_CMD="python grpo_prisoners_dilemma.py --config configs/grpo_prisoners_dilemma.yaml"
if [ ! -z "$CONFIG_OVERRIDES" ]; then
    PYTHON_CMD="$PYTHON_CMD $CONFIG_OVERRIDES"
fi
echo "Command: $PYTHON_CMD"
echo ""

# Run the training with output redirection to job-specific log
eval "$PYTHON_CMD" 2>&1 | tee "$JOB_LOG_DIR/training_output.log"

# Capture exit status
TRAINING_EXIT_CODE=$?

echo ""
echo "🏁 Training completed at $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"

# Post-training information
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "📊 Post-training GPU memory:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
else
    echo "📊 Post-training GPU memory: nvidia-smi not available"
fi

# Copy important files to job log directory
echo ""
echo "📋 Copying important files to log directory..."
cp configs/grpo_prisoners_dilemma.yaml "$JOB_LOG_DIR/" 2>/dev/null || echo "Config file not found"
cp -r checkpoints/grpo_prisoners_dilemma "$JOB_LOG_DIR/" 2>/dev/null || echo "Checkpoints not found"

# Copy wandb logs if they exist
if [ -d "wandb" ]; then
    echo "📈 Copying W&B logs..."
    cp -r wandb "$JOB_LOG_DIR/"
fi

# Copy any evaluation results
if [ -f "eval_results.json" ]; then
    echo "🧪 Copying evaluation results..."
    cp eval_results.json "$JOB_LOG_DIR/"
fi

# Copy GRPO-specific results
if [ -d "eval_results/grpo_prisoners_dilemma" ]; then
    echo "🧪 Copying GRPO evaluation results..."
    cp -r eval_results/grpo_prisoners_dilemma "$JOB_LOG_DIR/"
fi

# Copy GRPO outputs if they exist
if [ -d "outputs/grpo_prisoners_dilemma" ]; then
    echo "📊 Copying GRPO training outputs..."
    cp -r outputs/grpo_prisoners_dilemma "$JOB_LOG_DIR/"
fi

# Create summary file
echo ""
echo "📝 Creating job summary..."
cat > "$JOB_LOG_DIR/job_summary.txt" << EOF
GRPO Prisoner's Dilemma Training Job Summary
============================================

Job Details:
- Job ID: $SLURM_JOB_ID
- Node: $SLURMD_NODENAME
- GPU: $CUDA_VISIBLE_DEVICES
- Start Time: $(date -d @$SLURM_JOB_START_TIME 2>/dev/null || echo "Unknown")
- End Time: $(date)
- Exit Code: $TRAINING_EXIT_CODE
- W&B Project: $WANDB_PROJECT

Training Description:
- Algorithm: GRPO (Group Relative Policy Optimization)
- Game: Iterated Prisoner's Dilemma
- Self-play: Main model vs frozen opponent (refreshed every N steps)
- Features: WSLS bot opponents, action noise, CPU parallelism
- Model: Single LLM with LoRA adaptation

Environment:
- Python: $(python --version 2>&1)
- PyTorch: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not found")
- Transformers: $(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "Not found")
- TRL: $(python -c "import trl; print(trl.__version__)" 2>/dev/null || echo "Not found")

GPU Information:
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info unavailable")

Files in log directory:
$(ls -la "$JOB_LOG_DIR/")

Status: $([ $TRAINING_EXIT_CODE -eq 0 ] && echo "SUCCESS" || echo "FAILED")
EOF

echo "✅ Job summary saved to: $JOB_LOG_DIR/job_summary.txt"
echo ""
echo "📁 All outputs saved to: $JOB_LOG_DIR"
echo "🎉 SLURM job completed!"

exit $TRAINING_EXIT_CODE