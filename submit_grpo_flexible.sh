#!/bin/bash
# Flexible SLURM script for GRPO distributed training
# Usage: sbatch --nodes=N submit_grpo_flexible.sh [config_file]
# Where N can be 1, 2, 3, or 4 nodes

#SBATCH --job-name=grpo-flex
#SBATCH --output=slurm-grpo-flex-%j.out
#SBATCH --error=slurm-grpo-flex-%j.err
#SBATCH --ntasks-per-node=2         # 2 V100s per node
#SBATCH --gres=gpu:volta:2          # Request 2 V100 GPUs per node
#SBATCH --cpus-per-task=8           # CPU cores per task
#SBATCH --mem=64G                   # Memory per node
#SBATCH --time=12:00:00             # Max runtime
#SBATCH --partition=xeon-g6-volta   # MIT Supercloud partition

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

# Environment setup
echo "🔧 Setting up environment..."
cd $SLURM_SUBMIT_DIR

# Load necessary modules for Supercloud
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

# Launch training
# GPU and environment information
echo "🎮 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

echo "💾 Memory Information:"
free -h
echo ""

echo "📦 Environment Information:"
python --version
pip list | grep -E "(torch|transformers|trl|vllm|peft|datasets)"
echo ""

echo "🏃 Launching training on $WORLD_SIZE GPUs..."
echo "Command: srun python -u grpo_code_execution.py --config $CONFIG_FILE"
echo ""

# Run the training with output redirection to job-specific log
srun python -u grpo_code_execution.py --config "$CONFIG_FILE" \
    2>&1 | tee "$GRPO_LOG_DIR/training_output.log"

# Capture exit status
TRAINING_EXIT_CODE=$?

echo ""
echo "🏁 Training completed at $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"

# Post-training cleanup and logging
echo ""
echo "📊 Post-training GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# Copy important files to job log directory
echo ""
echo "📋 Copying important files to log directory..."
cp "$CONFIG_FILE" "$GRPO_LOG_DIR/" 2>/dev/null || echo "Config file not found"
cp -r checkpoints/grpo_code_execution "$GRPO_LOG_DIR/" 2>/dev/null || echo "Checkpoints not found"

# Copy wandb logs if they exist
if [ -d "wandb" ]; then
    echo "📈 Copying W&B logs..."
    cp -r wandb "$GRPO_LOG_DIR/"
fi

# Create summary file
echo ""
echo "📝 Creating job summary..."
cat > "$GRPO_LOG_DIR/job_summary.txt" << EOF
GRPO Flexible Distributed Training Job Summary
==============================================

Job Details:
- Job ID: $SLURM_JOB_ID
- Nodes: $SLURM_NNODES
- Total GPUs: $WORLD_SIZE
- Start Time: $(date -d @$SLURM_JOB_START_TIME 2>/dev/null || echo "Unknown")
- End Time: $(date)
- Exit Code: $TRAINING_EXIT_CODE

Distributed Setup:
- WORLD_SIZE: $WORLD_SIZE
- MASTER_ADDR: $MASTER_ADDR
- MASTER_PORT: $MASTER_PORT

Environment:
- Python: $(python --version 2>&1)
- PyTorch: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not found")
- Config: $CONFIG_FILE

Status: $([ $TRAINING_EXIT_CODE -eq 0 ] && echo "SUCCESS" || echo "FAILED")
EOF

echo "✅ Job summary saved to: $GRPO_LOG_DIR/job_summary.txt"
echo ""
echo "📁 All outputs saved to: $GRPO_LOG_DIR"
echo "🎉 Flexible distributed SLURM job completed!"

exit $TRAINING_EXIT_CODE