#!/bin/bash
# Smart launcher for GRPO training - automatically chooses single or distributed based on available GPUs

# Parse command line arguments
CONFIG_OVERRIDES=""
FORCE_MODE=""

echo "🎯 GRPO Training Launcher"
echo "========================"

# Parse arguments
for arg in "$@"; do
    if [[ $arg == "--single" ]]; then
        FORCE_MODE="single"
    elif [[ $arg == "--distributed" ]]; then
        FORCE_MODE="distributed"
    elif [[ $arg == --* ]]; then
        CONFIG_OVERRIDES="$CONFIG_OVERRIDES $arg"
    fi
done

# Detect available GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "🎮 Detected $GPU_COUNT GPU(s)"
else
    GPU_COUNT=0
    echo "⚠️ No NVIDIA GPUs detected"
fi

# Determine training mode
if [[ $FORCE_MODE == "single" ]]; then
    MODE="single"
    echo "🔧 Forced single GPU mode"
elif [[ $FORCE_MODE == "distributed" ]]; then
    MODE="distributed"
    echo "🔧 Forced distributed mode"
elif [[ $GPU_COUNT -ge 2 ]]; then
    MODE="distributed"
    echo "🚀 Auto-selected distributed mode (2+ GPUs available)"
else
    MODE="single"
    echo "📱 Auto-selected single GPU mode"
fi

# Launch appropriate training
if [[ $MODE == "distributed" ]]; then
    echo ""
    echo "🌐 Launching distributed training..."
    echo "Command: sbatch submit_grpo_training_distributed.sh$CONFIG_OVERRIDES"
    echo ""
    sbatch submit_grpo_training_distributed.sh $CONFIG_OVERRIDES
else
    echo ""
    echo "🎯 Launching single GPU training..."
    echo "Command: sbatch submit_grpo_training.sh$CONFIG_OVERRIDES"
    echo ""
    sbatch submit_grpo_training.sh $CONFIG_OVERRIDES
fi

echo ""
echo "📊 Monitor your job with:"
echo "   squeue -u \$USER"
echo "   watch nvidia-smi"
echo ""
echo "📁 Logs will be saved to:"
if [[ $MODE == "distributed" ]]; then
    echo "   logs/grpo_distributed_job_[JOB_ID]_[TIMESTAMP]/"
else
    echo "   logs/grpo_code_execute_job_[JOB_ID]_[TIMESTAMP]/"
fi