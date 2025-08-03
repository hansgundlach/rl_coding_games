#!/bin/bash
#SBATCH --job-name=grpo-code-game-icl
#SBATCH --partition=xeon-p8
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/job_%j_grpo_code_game_icl/training_output.log
#SBATCH --error=logs/job_%j_grpo_code_game_icl/training_error.log

echo "🚀 Starting GRPO Code Game with ICL Memory training job: $SLURM_JOB_ID"
echo "📅 Started at: $(date)"
echo "🖥️ Running on: $(hostname)"
echo "🎮 GPU: $CUDA_VISIBLE_DEVICES"

# Create job-specific log directory
export GRPO_LOG_DIR="logs/job_${SLURM_JOB_ID}_grpo_code_game_icl"
mkdir -p "$GRPO_LOG_DIR"

echo "📂 Logs will be saved to: $GRPO_LOG_DIR"

# Set environment variables for distributed training
export TOKENIZERS_PARALLELISM=false

# Load conda environment
echo "🐍 Loading conda environment..."
module load anaconda/2023a
source activate llm_training

echo "📍 Working directory: $(pwd)"
echo "🐍 Python path: $(which python)"
echo "🧠 GPU memory:"
nvidia-smi

# Parse command line arguments for config overrides
CONFIG_OVERRIDES=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --*)
            CONFIG_OVERRIDES="$CONFIG_OVERRIDES $1"
            if [[ $2 != --* ]] && [[ $# -gt 1 ]]; then
                CONFIG_OVERRIDES="$CONFIG_OVERRIDES $2"
                shift
            fi
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [[ -n "$CONFIG_OVERRIDES" ]]; then
    echo "🔧 Config overrides: $CONFIG_OVERRIDES"
fi

echo "🎯 Starting GRPO Code Game with ICL training..."

# Run the training script with config overrides
python grpo_code_game_icl.py --config configs/grpo_code_game_icl.yaml $CONFIG_OVERRIDES

echo "✅ Training completed at: $(date)"
echo "📊 Final GPU memory usage:"
nvidia-smi

# Copy important logs to job directory
if [[ -d "./eval_results" ]]; then
    cp -r ./eval_results/* "$GRPO_LOG_DIR/" 2>/dev/null || true
fi

if [[ -d "./checkpoints" ]]; then
    echo "💾 Checkpoint summary:"
    find ./checkpoints -name "*.bin" -o -name "*.safetensors" | head -5
fi

echo "🏁 GRPO Code Game with ICL job $SLURM_JOB_ID completed"