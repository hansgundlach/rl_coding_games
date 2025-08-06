#!/bin/bash
#SBATCH --job-name=unified_training
#SBATCH --output=logs/job_%j_%x/slurm-unified-%j.out
#SBATCH --error=logs/job_%j_%x/slurm-unified-%j.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# MIT Supercloud SLURM script for Unified Training
# Single V100 GPU training with logs output
#
# Usage: sbatch submit_unified_training.sh <python_script> <config_file> [config_overrides...]
# Examples:
#   sbatch submit_unified_training.sh grpo_code_execute.py configs/grpo_code_execution.yaml --training_args.max_steps=50
#   sbatch submit_unified_training.sh spiral_self_play.py configs/spiral_self_play.yaml --training.num_steps=50
#   sbatch submit_unified_training.sh grpo_code_game_icl.py configs/grpo_code_game_icl.yaml --training.num_steps=50

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "❌ Error: Missing required arguments"
    echo "Usage: sbatch submit_unified_training.sh <python_script> <config_file> [config_overrides...]"
    echo "Examples:"
    echo "  sbatch submit_unified_training.sh grpo_code_execute.py configs/grpo_code_execution.yaml"
    echo "  sbatch submit_unified_training.sh spiral_self_play.py configs/spiral_self_play.yaml --training.num_steps=50"
    echo "  sbatch submit_unified_training.sh grpo_code_game_icl.py configs/grpo_code_game_icl.yaml"
    exit 1
fi

# Extract script and config from arguments
PYTHON_SCRIPT="$1"
CONFIG_FILE="$2"
shift 2  # Remove script and config from arguments

# Validate script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Python script '$PYTHON_SCRIPT' not found"
    exit 1
fi

# Validate config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Extract script name for job naming
SCRIPT_NAME=$(basename "$PYTHON_SCRIPT" .py)
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)

# Parse remaining command line arguments for config overrides
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

echo "🚀 Starting Unified Training on Supercloud V100"
echo "==============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Script: $PYTHON_SCRIPT"
echo "Config: $CONFIG_FILE"
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
echo "�� Activating virtual environment..."
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

# Add W&B project override capability
if [ -z "$WANDB_PROJECT" ]; then
    # Extract project name from script name for consistency
    SCRIPT_BASE=$(echo "$SCRIPT_NAME" | sed 's/_/-/g')
    export WANDB_PROJECT="$SCRIPT_BASE"
    echo "📊 Set WANDB_PROJECT to: $WANDB_PROJECT"
else
    echo "📊 Using WANDB_PROJECT override: $WANDB_PROJECT"
fi

# Set additional environment variables for specific scripts
if [[ "$SCRIPT_NAME" == *"grpo_code_game_icl"* ]]; then
    echo "🔧 Setting ICL-specific environment variables..."
    export TOKENIZERS_PARALLELISM=false
fi

# Set up logs directory with job-specific subfolder
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_LOG_DIR="logs/${SCRIPT_NAME}_job_${SLURM_JOB_ID}_${TIMESTAMP}"
mkdir -p "$JOB_LOG_DIR"
echo "📁 Created job log directory: $JOB_LOG_DIR"

# Export log directory for Python script to use (script-specific)
if [[ "$SCRIPT_NAME" == *"grpo"* ]]; then
    export GRPO_LOG_DIR="$JOB_LOG_DIR"
elif [[ "$SCRIPT_NAME" == *"spiral"* ]]; then
    export SPIRAL_LOG_DIR="$JOB_LOG_DIR"
fi

# GPU information
echo "�� GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Memory information
echo "�� Memory Information:"
free -h
echo ""

# Python and package versions
echo "📦 Environment Information:"
python --version
pip list | grep -E "(torch|transformers|trl|vllm|peft|datasets)"
echo ""

# Start training with timestamping
echo "🏋️ Starting $SCRIPT_NAME training at $(date)..."
PYTHON_CMD="python $PYTHON_SCRIPT --config $CONFIG_FILE"
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
echo "�� Training completed at $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"

# Post-training information
echo ""
echo "📊 Post-training GPU memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# Copy important files to job log directory
echo ""
echo "📋 Copying important files to log directory..."
cp "$CONFIG_FILE" "$JOB_LOG_DIR/" 2>/dev/null || echo "Config file not found"

# Copy checkpoints based on script type
if [[ "$SCRIPT_NAME" == *"grpo_code_execute"* ]]; then
    cp -r checkpoints/grpo_code_execution "$JOB_LOG_DIR/" 2>/dev/null || echo "Checkpoints not found"
elif [[ "$SCRIPT_NAME" == *"grpo_code_game_icl"* ]]; then
    cp -r checkpoints/grpo_code_game_icl "$JOB_LOG_DIR/" 2>/dev/null || echo "Checkpoints not found"
elif [[ "$SCRIPT_NAME" == *"spiral_self_play"* ]]; then
    cp -r checkpoints/spiral_self_play "$JOB_LOG_DIR/" 2>/dev/null || echo "Checkpoints not found"
elif [[ "$SCRIPT_NAME" == *"spiral_code_game"* ]]; then
    cp -r checkpoints/spiral_code_game "$JOB_LOG_DIR/" 2>/dev/null || echo "Checkpoints not found"
elif [[ "$SCRIPT_NAME" == *"spiral_prisoners_dilemma"* ]]; then
    cp -r checkpoints/spiral_prisoners_dilemma "$JOB_LOG_DIR/" 2>/dev/null || echo "Checkpoints not found"
fi

# Copy wandb logs if they exist
if [ -d "wandb" ]; then
    echo "📈 Copying W&B logs..."
    cp -r wandb "$JOB_LOG_DIR/"
fi

# Copy any evaluation results
if [ -f "eval_results.json" ]; then
    echo "�� Copying evaluation results..."
    cp eval_results.json "$JOB_LOG_DIR/"
fi

# Copy script-specific evaluation results
if [ -d "eval_results/$SCRIPT_NAME" ]; then
    echo "🧪 Copying $SCRIPT_NAME evaluation results..."
    cp -r eval_results/$SCRIPT_NAME "$JOB_LOG_DIR/"
fi

# Create summary file
echo ""
echo "📝 Creating job summary..."
cat > "$JOB_LOG_DIR/job_summary.txt" << EOF
Unified Training Job Summary
===========================

Job Details:
- Job ID: $SLURM_JOB_ID
- Node: $SLURMD_NODENAME
- GPU: $CUDA_VISIBLE_DEVICES
- Script: $PYTHON_SCRIPT
- Config: $CONFIG_FILE
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