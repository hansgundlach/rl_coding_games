#!/bin/bash
# Flexible SLURM script for GRPO distributed training
# Usage: sbatch --nodes=N submit_grpo_flexible.sh [config_file]
# Where N can be 1, 2, 3, or 4 nodes

#SBATCH --job-name=grpo-flex
#SBATCH --output=logs/grpo_flex_%j.out
#SBATCH --error=logs/grpo_flex_%j.err
#SBATCH --ntasks-per-node=2         # 2 V100s per node
#SBATCH --gres=gpu:2                # Request 2 GPUs per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --mem=64G                   # Memory per node
#SBATCH --time=04:00:00             # Max runtime
#SBATCH --partition=xeon-v100       # MIT Supercloud partition

# Get config file from command line argument or use default
CONFIG_FILE=${1:-configs/grpo_code_execution.yaml}

# Validate nodes count
if [[ $SLURM_NNODES -lt 1 || $SLURM_NNODES -gt 4 ]]; then
    echo "❌ Error: Only 1-4 nodes supported. Got $SLURM_NNODES nodes."
    exit 1
fi

echo "🚀 Starting Flexible GRPO Distributed Training"
echo "📊 Job Configuration:"
echo "   - Nodes: $SLURM_NNODES"
echo "   - GPUs per node: $SLURM_NTASKS_PER_NODE"  
echo "   - Total GPUs: $((SLURM_NNODES * SLURM_NTASKS_PER_NODE))"
echo "   - Config file: $CONFIG_FILE"
echo "   - Job ID: $SLURM_JOB_ID"
echo ""

# Create logs directory
mkdir -p logs/job_${SLURM_JOB_ID}
export GRPO_LOG_DIR="logs/job_${SLURM_JOB_ID}"

# Set distributed training environment
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355

echo "🌐 Distributed Setup:"
echo "   - WORLD_SIZE: $WORLD_SIZE"
echo "   - MASTER_ADDR: $MASTER_ADDR"
echo "   - MASTER_PORT: $MASTER_PORT"
echo "   - Log directory: $GRPO_LOG_DIR"

# Performance recommendations based on GPU count
if [[ $WORLD_SIZE -eq 2 ]]; then
    echo "💡 Single node (2 GPUs): Good for testing and small experiments"
elif [[ $WORLD_SIZE -eq 4 ]]; then
    echo "💡 Two nodes (4 GPUs): Balanced performance and communication overhead"
elif [[ $WORLD_SIZE -eq 6 ]]; then
    echo "💡 Three nodes (6 GPUs): High performance with moderate communication"
elif [[ $WORLD_SIZE -eq 8 ]]; then
    echo "💡 Four nodes (8 GPUs): Maximum performance with higher communication overhead"
fi

echo ""

# Load required modules
module load python/3.9
module load cuda/11.8

# Launch training
echo "🏃 Launching training on $WORLD_SIZE GPUs..."
srun python -u grpo_code_execution.py --config "$CONFIG_FILE" \
    2>&1 | tee "$GRPO_LOG_DIR/training_output.log"

# Report completion
TRAINING_EXIT_CODE=$?
if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
fi

echo "📊 Results:"
echo "   - Logs: $GRPO_LOG_DIR"
echo "   - Training output: $GRPO_LOG_DIR/training_output.log"
echo "   - SLURM output: logs/grpo_flex_${SLURM_JOB_ID}.out"

exit $TRAINING_EXIT_CODE