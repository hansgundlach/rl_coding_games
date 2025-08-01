#!/bin/bash
#SBATCH --job-name=test-dist
#SBATCH --output=slurm-test-dist-%j.out
#SBATCH --error=slurm-test-dist-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=xeon-g6-volta

echo "🧪 Testing Distributed Setup"
echo "=========================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo ""

# Environment setup
cd $SLURM_SUBMIT_DIR
module purge
module load cuda/12.1
module load python/3.9
source venv/bin/activate

# Set distributed environment
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12355

echo "🌐 Distributed Environment:"
echo "   WORLD_SIZE: $WORLD_SIZE"
echo "   MASTER_ADDR: $MASTER_ADDR"
echo "   MASTER_PORT: $MASTER_PORT"
echo ""

# Test the distributed setup
srun --export=ALL \
    bash -c '
        export RANK=$SLURM_PROCID
        export LOCAL_RANK=$SLURM_LOCALID
        echo "Process $SLURM_PROCID on node $(hostname): RANK=$RANK, LOCAL_RANK=$LOCAL_RANK"
        python test_distributed.py
    '

echo "✅ Test completed!"