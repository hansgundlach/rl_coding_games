#!/bin/bash
#SBATCH --job-name=spiral_self_play
#SBATCH --output=logs/job_%j_%x/slurm-spiral-%j.out
#SBATCH --error=logs/job_%j_%x/slurm-spiral-%j.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# MIT Supercloud SLURM script for SPIRAL Self-Play Training
# Single V100 GPU training with logs output
#
# Usage: sbatch submit_spiral_training.sh [config_overrides...]
# Examples:
#   sbatch submit_spiral_training.sh --num_steps=50 --learning_rate=0.0001
#   sbatch submit_spiral_training.sh --games_per_step_v100=3 --eval_interval_steps=10

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

echo "🌀 Starting SPIRAL Self-Play Training on Supercloud V100"
echo "========================================================="
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
cd $SLURM_SUBMIT_DIR

# Load necessary modules (adjust for Supercloud)
module purge
module load cuda/12.1
module load python/3.9

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate || {
    echo "❌ Failed to activate virtual environment"
    echo "Please ensure you have created a virtual environment:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
}

# Set offline environment variables for Supercloud
echo "🌐 Setting offline mode for Supercloud..."
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

# Set up logs directory with job-specific subfolder
JOB_LOG_DIR="logs/spiral_job_${SLURM_JOB_ID}"
mkdir -p "$JOB_LOG_DIR"
echo "📁 Created job log directory: $JOB_LOG_DIR"

# Export log directory for Python script to use
export SPIRAL_LOG_DIR="$JOB_LOG_DIR"

# GPU information
echo "🎮 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Memory information
echo "💾 Memory Information:"
free -h
echo ""

# Python and package versions
echo "📦 Environment Information:"
python --version
pip list | grep -E "(torch|transformers|trl|vllm|peft|datasets)"
echo ""

# Start training with timestamping
echo "🏋️ Starting SPIRAL self-play training at $(date)..."
PYTHON_CMD="python spiral_self_play.py --config configs/spiral_self_play.yaml"
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
echo "📊 Post-training GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# Copy important files to job log directory
echo ""
echo "📋 Copying important files to log directory..."
cp configs/spiral_self_play.yaml "$JOB_LOG_DIR/" 2>/dev/null || echo "Config file not found"
cp -r checkpoints/spiral_self_play "$JOB_LOG_DIR/" 2>/dev/null || echo "Checkpoints not found"

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

# Copy SPIRAL-specific results
if [ -d "eval_results/spiral_self_play" ]; then
    echo "🧪 Copying SPIRAL evaluation results..."
    cp -r eval_results/spiral_self_play "$JOB_LOG_DIR/"
fi

# Create summary file
echo ""
echo "📝 Creating job summary..."
cat > "$JOB_LOG_DIR/job_summary.txt" << EOF
SPIRAL Self-Play Training Job Summary
=====================================

Job Details:
- Job ID: $SLURM_JOB_ID
- Node: $SLURMD_NODENAME
- GPU: $CUDA_VISIBLE_DEVICES
- Start Time: $(date -d @$SLURM_JOB_START_TIME 2>/dev/null || echo "Unknown")
- End Time: $(date)
- Exit Code: $TRAINING_EXIT_CODE

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