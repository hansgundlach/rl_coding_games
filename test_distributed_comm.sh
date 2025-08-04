#!/bin/bash
# SLURM script to test internode communication for distributed training
# Usage: sbatch --nodes=2 test_distributed_comm.sh

#SBATCH --job-name=test-dist-comm
#SBATCH --output=logs/test_dist_comm_%j.out
#SBATCH --error=logs/test_dist_comm_%j.err
#SBATCH --nodes=2                   # Exactly 2 nodes for internode test
#SBATCH --ntasks-per-node=2         # 2 V100s per node
#SBATCH --gres=gpu:volta:2          # Request 2 V100 GPUs per node
#SBATCH --cpus-per-task=4           # CPU cores per task
#SBATCH --mem=32G                   # Memory per node
#SBATCH --time=00:15:00             # 15 minutes should be enough
#SBATCH --partition=xeon-g6-volta   # MIT Supercloud partition

echo "🧪 INTERNODE COMMUNICATION TEST"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_NTASKS_PER_NODE"
echo "Total GPUs: $((SLURM_NNODES * SLURM_NTASKS_PER_NODE))"
echo "Start time: $(date)"
echo ""

# Validate we have exactly 2 nodes
if [[ $SLURM_NNODES -ne 2 ]]; then
    echo "❌ Error: This test requires exactly 2 nodes. Got $SLURM_NNODES nodes."
    echo "Usage: sbatch --nodes=2 test_distributed_comm.sh"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Set distributed training environment
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355
export NCCL_DEBUG=INFO

echo "🌐 Distributed Setup:"
echo "   WORLD_SIZE: $WORLD_SIZE"
echo "   MASTER_ADDR: $MASTER_ADDR"
echo "   MASTER_PORT: $MASTER_PORT"
echo ""

# Auto-detect network interface
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
    echo "  pip install torch"
    exit 1
}

# Set offline mode for Supercloud
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

echo ""

# Show node information
echo "📍 Node Information:"
scontrol show hostnames "$SLURM_JOB_NODELIST" | while read node; do
    echo "   - $node"
done
echo ""

# GPU information
echo "🎮 GPU Information on master node:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
echo ""

# Test basic connectivity between nodes
echo "🔗 Testing basic connectivity between nodes..."
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
OTHER_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2)

for node in $OTHER_NODES; do
    echo "   Testing connection to $node..."
    if ping -c 1 -W 2 "$node" > /dev/null 2>&1; then
        echo "   ✅ $node is reachable"
    else
        echo "   ❌ $node is NOT reachable"
    fi
done
echo ""

# Create wrapper script for distributed test
cat > test_wrapper.sh << 'EOF'
#!/bin/bash

# Set PyTorch distributed environment variables from SLURM
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Starting process $SLURM_PROCID on $(hostname) at $(date)"
echo "RANK=$RANK, LOCAL_RANK=$LOCAL_RANK, WORLD_SIZE=$WORLD_SIZE"

# Run the test
python -u test_distributed_comm.py
EOF

chmod +x test_wrapper.sh

echo "🏃 Launching distributed communication test..."
echo "================================================"

# Run the test
srun ./test_wrapper.sh

# Capture exit status
TEST_EXIT_CODE=$?

echo ""
echo "================================================"
echo "🏁 Test completed at $(date)"
echo "Exit code: $TEST_EXIT_CODE"

# Final summary
if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "🎉 SUCCESS! Internode communication test PASSED"
    echo "✅ Distributed training should work on this cluster"
    echo "✅ NCCL communication is functional between nodes"
    echo "🚀 You can proceed with distributed training!"
else
    echo ""
    echo "💥 FAILURE! Internode communication test FAILED"
    echo "❌ Distributed training may not work properly"
    echo "❌ Check network configuration and NCCL settings"
    echo ""
    echo "🔍 Troubleshooting tips:"
    echo "   1. Check if NCCL is properly installed"
    echo "   2. Verify network connectivity between nodes"
    echo "   3. Check firewall settings for the master port"
    echo "   4. Try a different MASTER_PORT if 12355 is blocked"
fi

# Cleanup
rm -f test_wrapper.sh

echo ""
echo "📄 Full output saved to: logs/test_dist_comm_${SLURM_JOB_ID}.out"
echo "📄 Error log saved to: logs/test_dist_comm_${SLURM_JOB_ID}.err"

exit $TEST_EXIT_CODE