#!/bin/bash
#SBATCH --job-name=grpo_code_game_icl_2gpu
#SBATCH --output=logs/job_%j_%x/slurm-%x-%j.out
#SBATCH --error=logs/job_%j_%x/slurm-%x-%j.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# MIT Supercloud SLURM script for GRPO Code Game with ICL – 2x V100 (single node)
# Launches with torchrun for DDP across 2 GPUs; forwards any config overrides.
#
# Usage: sbatch submit_grpo_code_game_icl_2gpu.sh [--key=value overrides]
# Examples:
#   sbatch submit_grpo_code_game_icl_2gpu.sh --training.num_steps=50 --icl.refresh_every=10
#   sbatch submit_grpo_code_game_icl_2gpu.sh --evaluation.num_questions=10 --training.games_per_step=4

# Parse command line arguments for config overrides
CONFIG_OVERRIDES=""
echo "🔍 Debug: Parsing arguments: $@"
for arg in "$@"; do
    echo "🔍 Debug: Processing argument: '$arg'"
    if [[ $arg == --* ]]; then
        CONFIG_OVERRIDES="$CONFIG_OVERRIDES $arg"
        echo "🔍 Debug: Added override: $arg"
    fi
done
echo "🔍 Debug: Final CONFIG_OVERRIDES: '$CONFIG_OVERRIDES'"

echo "🚀 Starting GRPO Code Game with ICL (2×GPU) on Supercloud V100"
echo "=================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
if [ ! -z "$CONFIG_OVERRIDES" ]; then
    echo "Config overrides: $CONFIG_OVERRIDES"
fi
echo ""

# Environment setup
echo "🔧 Setting up environment..."
cd "$SLURM_SUBMIT_DIR"

# Load modules
module purge
module load cuda/12.1
module load python/3.9

# Activate virtual environment
echo "🐍 Activating virtual environment..."
source venv/bin/activate || {
    echo "❌ Failed to activate virtual environment"
    echo "Create it and install deps:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
}

# Offline-friendly defaults for Supercloud
echo "🌐 Setting offline mode for Supercloud..."
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

# W&B project (consistent naming)
if [ -z "$WANDB_PROJECT" ]; then
    export WANDB_PROJECT="grpo-code-game-icl"
    echo "📊 Set WANDB_PROJECT to: $WANDB_PROJECT"
else
    echo "📊 Using WANDB_PROJECT override: $WANDB_PROJECT"
fi

# Logs directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_LOG_DIR="logs/grpo_code_game_icl_2gpu_job_${SLURM_JOB_ID}_${TIMESTAMP}"
mkdir -p "$JOB_LOG_DIR"
echo "📁 Created job log directory: $JOB_LOG_DIR"
export GRPO_LOG_DIR="$JOB_LOG_DIR"

# System info
echo "🎮 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || true
echo ""
echo "💾 Memory Information:"
free -h || true
echo ""
echo "📦 Environment Information:"
python --version
pip list | grep -E "(torch|transformers|trl|vllm|peft|datasets)" || true
echo ""

# Determine number of processes per node (default 2)
NPROC=${SLURM_GPUS_ON_NODE:-2}
if [ -z "$NPROC" ] || [ "$NPROC" -lt 1 ]; then
    NPROC=2
fi
echo "🧮 Using nproc-per-node=$NPROC"

# Start training
echo "🏋️ Starting GRPO Code Game with ICL training at $(date)..."
CMD="torchrun --nproc-per-node=$NPROC grpo_code_game_icl.py --config configs/grpo_code_game_icl.yaml --training.num_gpus=$NPROC"
if [ ! -z "$CONFIG_OVERRIDES" ]; then
    CMD="$CMD $CONFIG_OVERRIDES"
fi
echo "Command: $CMD"
echo ""

# Run and tee output
eval "$CMD" 2>&1 | tee "$JOB_LOG_DIR/training_output.log"
TRAINING_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "🏁 Training completed at $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"

# Post-run info
echo ""
echo "📊 Post-training GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits || true

# Copy artifacts
echo ""
echo "📋 Copying important files to log directory..."
cp configs/grpo_code_game_icl.yaml "$JOB_LOG_DIR/" 2>/dev/null || echo "Config file not found"
cp -r checkpoints/grpo_code_game_icl "$JOB_LOG_DIR/" 2>/dev/null || echo "Checkpoints not found"
if [ -d "wandb" ]; then
    echo "📈 Copying W&B logs..."
    cp -r wandb "$JOB_LOG_DIR/"
fi
if [ -f "eval_results.json" ]; then
    echo "🧪 Copying evaluation results..."
    cp eval_results.json "$JOB_LOG_DIR/"
fi
if [ -d "eval_results/grpo_code_game_icl" ]; then
    echo "🧪 Copying ICL evaluation results..."
    cp -r eval_results/grpo_code_game_icl "$JOB_LOG_DIR/"
fi

# Summary
echo ""
echo "📝 Creating job summary..."
cat > "$JOB_LOG_DIR/job_summary.txt" << EOF
GRPO Code Game with ICL Training (2×GPU) Job Summary
====================================================

Job Details:
- Job ID: $SLURM_JOB_ID
- Node: $SLURMD_NODENAME
- GPUs: $CUDA_VISIBLE_DEVICES
- Start Time: 
- End Time: $(date)
- Exit Code: $TRAINING_EXIT_CODE
- W&B Project: $WANDB_PROJECT

Environment:
- Python: $(python --version 2>&1)
- PyTorch: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not found")
- Transformers: $(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "Not found")
- TRL: $(python -c "import trl; print(trl.__version__)" 2>/dev/null || echo "Not found")

Artifacts saved in: $JOB_LOG_DIR
EOF

echo "✅ Job summary saved to: $JOB_LOG_DIR/job_summary.txt"
echo "📁 All outputs saved to: $JOB_LOG_DIR"
echo "🎉 SLURM job completed!"

exit $TRAINING_EXIT_CODE


