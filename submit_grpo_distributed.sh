#!/bin/bash
#SBATCH --job-name=grpo-distributed
#SBATCH --output=logs/grpo_distributed_%j.out
#SBATCH --error=logs/grpo_distributed_%j.err
#SBATCH --nodes=4                    # Number of nodes (adjust as needed: 1-4)
#SBATCH --ntasks-per-node=2         # Number of GPUs per node (2 V100s per node)
#SBATCH --gres=gpu:2                # Request 2 GPUs per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --mem=64G                   # Memory per node
#SBATCH --time=04:00:00             # Max runtime (4 hours)
#SBATCH --partition=xeon-v100       # MIT Supercloud V100 partition

# Configuration variables (can be overridden via environment)
NODES=${SLURM_NNODES:-4}
GPUS_PER_NODE=${SLURM_NTASKS_PER_NODE:-2}
CONFIG_FILE=${CONFIG_FILE:-configs/grpo_code_execution.yaml}

echo "🚀 Starting GRPO Distributed Training"
echo "📊 Configuration:"
echo "   - Nodes: $NODES"
echo "   - GPUs per node: $GPUS_PER_NODE"  
echo "   - Total GPUs: $((NODES * GPUS_PER_NODE))"
echo "   - Config file: $CONFIG_FILE"
echo "   - Job ID: $SLURM_JOB_ID"

# Create logs directory if it doesn't exist
mkdir -p logs/job_${SLURM_JOB_ID}

# Set log directory for the training script
export GRPO_LOG_DIR="logs/job_${SLURM_JOB_ID}"

# Set distributed training environment variables
export WORLD_SIZE=$((NODES * GPUS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355

# Print distributed training info
echo "🌐 Distributed Training Setup:"
echo "   - WORLD_SIZE: $WORLD_SIZE"
echo "   - MASTER_ADDR: $MASTER_ADDR"
echo "   - MASTER_PORT: $MASTER_PORT"
echo "   - Log directory: $GRPO_LOG_DIR"

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up distributed processes..."
    pkill -f "grpo_code_execution.py" || true
}
trap cleanup EXIT

# Load modules (adjust for your system)
module load python/3.9
module load cuda/11.8

# Activate virtual environment if needed
# source venv/bin/activate

echo "🏃 Launching distributed training..."

# Use srun to launch the distributed training
srun python -u grpo_code_execution.py \
    --config "$CONFIG_FILE" \
    2>&1 | tee "$GRPO_LOG_DIR/training_output.log"

echo "✅ Training completed!"
echo "📊 Logs saved to: $GRPO_LOG_DIR"
echo "🎯 Check training_output.log for detailed output"