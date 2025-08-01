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

# Create logs directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_LOG_DIR="logs/job_${SLURM_JOB_ID}_${TIMESTAMP}"
mkdir -p "$JOB_LOG_DIR"
echo "📁 Created job log directory: $JOB_LOG_DIR"
echo "🕒 Job started at: $(date)"
export GRPO_LOG_DIR="$JOB_LOG_DIR"

# Set distributed training environment
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355
export NCCL_DEBUG=INFO  # Enable NCCL debugging

# MIT Supercloud specific: Set NCCL network interface for high-speed interconnect
# This is critical for multi-node communication on HPC clusters.
# Auto-detect available network interface
if ip link show ib0 >/dev/null 2>&1; then
    export NCCL_SOCKET_IFNAME=ib0
    echo "🌐 Using InfiniBand interface: ib0"
elif ip link show eth0 >/dev/null 2>&1; then
    export NCCL_SOCKET_IFNAME=eth0
    echo "🌐 Using Ethernet interface: eth0"
elif ip link show eno1 >/dev/null 2>&1; then
    export NCCL_SOCKET_IFNAME=eno1
    echo "🌐 Using Ethernet interface: eno1"
else
    echo "⚠️ No known network interface found, using default NCCL settings"
    echo "Available interfaces:"
    ip link show | grep -E "^[0-9]+:" | cut -d: -f2 | tr -d ' '
fi

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

# Function to cleanup on exit
cleanup() {
    echo "🧹 Cleaning up distributed processes..."
    pkill -f "grpo_code_execution.py" || true
}
trap cleanup EXIT

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

# Create a wrapper script for each process
cat > "$GRPO_LOG_DIR/run_process.sh" << 'EOF'
#!/bin/bash
set -e

# Log process start with timestamp
echo "🕒 Process $SLURM_PROCID started at: $(date)" 
echo "🌐 Hostname: $(hostname)"
echo "📍 Working directory: $(pwd)"
echo "🔍 Node ID: $SLURM_NODEID, Local ID: $SLURM_LOCALID"
echo "🎮 GPU devices: $CUDA_VISIBLE_DEVICES"
echo ""

# Set PyTorch distributed environment variables from SLURM
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Debug GPU assignment
echo "🔍 GPU debugging for process $SLURM_PROCID:"
echo "   Node: $(hostname)"
echo "   SLURM_NODEID: $SLURM_NODEID"
echo "   SLURM_LOCALID: $SLURM_LOCALID" 
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if command -v nvidia-smi &> /dev/null; then
    echo "   Available GPUs on this node:"
    nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits | while read line; do
        echo "     $line"
    done
else
    echo "   nvidia-smi not available"
fi

echo "🚀 Distributed environment for process $SLURM_PROCID:"
echo "   RANK=$RANK"
echo "   LOCAL_RANK=$LOCAL_RANK" 
echo "   WORLD_SIZE=$WORLD_SIZE"
echo "   MASTER_ADDR=$MASTER_ADDR"
echo "   MASTER_PORT=$MASTER_PORT"
echo ""

# Verify environment variables are set
if [ -z "$RANK" ] || [ -z "$LOCAL_RANK" ] || [ -z "$WORLD_SIZE" ]; then
    echo "❌ Error: Required environment variables not set!"
    echo "   RANK: '$RANK'"
    echo "   LOCAL_RANK: '$LOCAL_RANK'"
    echo "   WORLD_SIZE: '$WORLD_SIZE'"
    exit 1
fi

# Verify LOCAL_RANK makes sense for available GPUs
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_VISIBLE_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    if [ "$LOCAL_RANK" -ge "$NUM_VISIBLE_GPUS" ]; then
        echo "❌ Error: LOCAL_RANK ($LOCAL_RANK) >= available GPUs ($NUM_VISIBLE_GPUS)"
        echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        echo "   This indicates a SLURM GPU assignment issue"
        exit 1
    fi
fi

echo "🏃 Starting Python training at $(date)..."
python -u grpo_code_execution.py --config "$1"
EOF

chmod +x "$GRPO_LOG_DIR/run_process.sh"

# Run training with timestamped log file
LOG_FILE="$GRPO_LOG_DIR/training_output_$(date +%Y%m%d_%H%M%S).log"
echo "📄 Training output will be logged to: $LOG_FILE"

srun "$GRPO_LOG_DIR/run_process.sh" "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

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