#!/bin/bash
#SBATCH --job-name=grpo_distributed
#SBATCH --output=logs/grpo_distributed_job_%j_%x/slurm-grpo-%j.out
#SBATCH --error=logs/grpo_distributed_job_%j_%x/slurm-grpo-%j.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# MIT Supercloud SLURM script for Distributed GRPO Code Execution Training
# Dual V100 GPU distributed training for maximum speed
#
# Usage: sbatch submit_grpo_training_distributed.sh [config_overrides...]
# Examples:
#   sbatch submit_grpo_training_distributed.sh --training_args.max_steps=50 --training_args.learning_rate=0.0001
#   sbatch submit_grpo_training_distributed.sh --training_args.per_device_train_batch_size=4 --evaluation.eval_interval_steps=10

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

echo "🚀 Starting Distributed GRPO Code Execution Training on Supercloud V100s"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
if [ ! -z "$CONFIG_OVERRIDES" ]; then
    echo "Config overrides: $CONFIG_OVERRIDES"
fi
echo ""

# Environment setup
echo "🔧 Setting up environment..."
cd $SLURM_SUBMIT_DIR

# Load necessary modules
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
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_LOG_DIR="logs/grpo_distributed_job_${SLURM_JOB_ID}_${TIMESTAMP}"
mkdir -p "$JOB_LOG_DIR"
echo "📁 Created job log directory: $JOB_LOG_DIR"

# Export log directory for Python script to use
export GRPO_LOG_DIR="$JOB_LOG_DIR"

# Set up distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=2
export NCCL_DEBUG=INFO

# Show distributed setup
echo "🌐 Distributed Training Setup:"
echo "   WORLD_SIZE: $WORLD_SIZE"
echo "   MASTER_ADDR: $MASTER_ADDR"
echo "   MASTER_PORT: $MASTER_PORT"
echo ""

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

# Verify PyTorch distributed is available
echo "🔍 Verifying PyTorch distributed support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); import torch.distributed as dist; print('Distributed support: Available')" || {
    echo "❌ PyTorch distributed support not available"
    exit 1
}
echo ""

# Start distributed training with torchrun
echo "🏋️ Starting distributed GRPO training at $(date)..."
echo "Using torchrun for distributed training across 2 V100 GPUs"

# Build torchrun command
TORCHRUN_CMD="torchrun"
TORCHRUN_CMD="$TORCHRUN_CMD --standalone"
TORCHRUN_CMD="$TORCHRUN_CMD --nnodes=1"
TORCHRUN_CMD="$TORCHRUN_CMD --nproc_per_node=2"
TORCHRUN_CMD="$TORCHRUN_CMD --master_port=$MASTER_PORT"
TORCHRUN_CMD="$TORCHRUN_CMD grpo_code_execute_distributed.py"
TORCHRUN_CMD="$TORCHRUN_CMD --config configs/grpo_code_execution.yaml"

if [ ! -z "$CONFIG_OVERRIDES" ]; then
    TORCHRUN_CMD="$TORCHRUN_CMD $CONFIG_OVERRIDES"
fi

echo "Command: $TORCHRUN_CMD"
echo ""

# Function to cleanup background processes
cleanup() {
    echo "🧹 Cleaning up distributed processes..."
    pkill -f "grpo_code_execute_distributed.py" || true
    sleep 2
}
trap cleanup EXIT

# Run the distributed training
eval "$TORCHRUN_CMD" 2>&1 | tee "$JOB_LOG_DIR/training_output.log"

# Capture exit status
TRAINING_EXIT_CODE=$?

echo ""
echo "🏁 Distributed training completed at $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"

# Post-training cleanup and logging
echo ""
echo "📊 Post-training GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# Copy important files to job log directory
echo ""
echo "📋 Copying important files to log directory..."
cp configs/grpo_code_execution.yaml "$JOB_LOG_DIR/" 2>/dev/null || echo "Config file not found"
cp -r checkpoints/grpo_code_execution "$JOB_LOG_DIR/" 2>/dev/null || echo "Checkpoints not found"

# Copy wandb logs if they exist
if [ -d "wandb" ]; then
    echo "📈 Copying W&B logs..."
    cp -r wandb "$JOB_LOG_DIR/"
fi

# Copy any evaluation results
if [ -d "eval_results" ]; then
    echo "🧪 Copying evaluation results..."
    cp -r eval_results "$JOB_LOG_DIR/"
fi

# Create summary file
echo ""
echo "📝 Creating job summary..."
cat > "$JOB_LOG_DIR/job_summary.txt" << EOF
Distributed GRPO Code Execution Training Job Summary
===================================================

Job Details:
- Job ID: $SLURM_JOB_ID
- Node: $SLURMD_NODENAME
- GPUs: $CUDA_VISIBLE_DEVICES (2 V100s)
- Start Time: $(date -d @$SLURM_JOB_START_TIME 2>/dev/null || echo "Unknown")
- End Time: $(date)
- Exit Code: $TRAINING_EXIT_CODE

Distributed Setup:
- World Size: $WORLD_SIZE
- Master Address: $MASTER_ADDR
- Master Port: $MASTER_PORT

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
echo "🎉 Distributed SLURM job completed!"

exit $TRAINING_EXIT_CODE