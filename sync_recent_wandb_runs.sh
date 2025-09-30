#!/bin/bash

# Script to sync the N most recent offline wandb runs
# Updated to work with consistent project naming

# Default to 5 runs if no parameter provided
N_RUNS=${1:-5}

# Validate input
if ! [[ "$N_RUNS" =~ ^[0-9]+$ ]] || [ "$N_RUNS" -lt 1 ]; then
    echo "❌ Error: Please provide a positive number of runs to sync"
    echo "Usage: $0 [number_of_runs]"
    echo "Example: $0 10    # Sync the 10 most recent runs"
    echo "Example: $0        # Sync the 5 most recent runs (default)"
    exit 1
fi

echo "🔄 Finding and syncing the $N_RUNS most recent offline wandb runs..."
echo

# Check if wandb directory exists
if [ ! -d "wandb" ]; then
    echo "❌ Error: wandb directory not found in current directory"
    exit 1
fi

# Change to wandb directory
cd wandb

# Find all offline-run directories, sort by modification time (newest first) and take the first N
recent_runs=$(find . -maxdepth 1 -type d -name "offline-run-*" -printf '%T@ %p\n' | sort -rn | head -$N_RUNS | cut -d' ' -f2-)

# Check if any runs were found
if [ -z "$recent_runs" ]; then
    echo "❌ No offline wandb runs found in wandb/ directory"
    echo "Looking for directories with pattern: offline-run-*"
    echo
    echo "📁 Available directories:"
    ls -la | grep "^d" | head -10
    exit 1
fi

echo "📋 Found the following recent offline runs:"
echo "$recent_runs" | sed 's|^\./||' | nl -w2 -s'. '
echo
#stuff

# Show project mapping info
echo "📊 Project Mapping Information:"
echo "   - Each run will be synced to its respective project"
echo "   - Project names are determined by the training script used"
echo "   - Run names will include timestamps for uniqueness"
echo "   - All runs from grpo_code_execute.py will go to 'qwen-code-execution-grpo'"
echo "   - All runs from grpo_code_game_icl.py will go to 'grpo-code-game-icl'"
echo "   - New runs should have readable names like 'grpo-code-execute-Aug06_2025_13h41m'"
echo

# Sync each run
count=0
for run_dir in $recent_runs; do
    count=$((count + 1))
    run_name=$(basename "$run_dir")
    echo "🚀 [$count/$N_RUNS] Syncing: $run_name"
    
    # Check if this is a new readable format or old cryptic format
    if [[ "$run_name" == *"grpo-code-execute-"* ]] || [[ "$run_name" == *"grpo-code-game-icl-"* ]]; then
        echo "   ✅ New readable format detected"
    else
        echo "   ⚠️ Old cryptic format detected (this should change with new runs)"
    fi
    
    # Show what project this will sync to (if we can determine it)
    if [ -f "$run_dir/files/config.yaml" ]; then
        project_name=$(grep -A5 "wandb:" "$run_dir/files/config.yaml" | grep "project_name_prefix:" | head -1 | sed 's/.*project_name_prefix:[[:space:]]*//' | tr -d '"')
        if [ ! -z "$project_name" ]; then
            echo "   📍 Will sync to project: $project_name"
        fi
    fi
    
    if wandb sync "$run_name"; then
        echo "✅ Successfully synced: $run_name"
    else
        echo "❌ Failed to sync: $run_name"
    fi
    echo
done

echo "🎉 Finished syncing $N_RUNS offline wandb runs!"
echo
echo "💡 Tips:"
echo "   - All runs from grpo_code_execute.py will sync to 'qwen-code-execution-grpo'"
echo "   - Each run will have a unique timestamp-based name like 'grpo-code-execute-Jul31_2025_14h30m'"
echo "   - You can view all runs in the same project on wandb.ai"
echo "   - Multiple training runs will appear as separate runs in the same project"
echo
echo "📝 Usage: $0 [number_of_runs]"
echo "   Example: $0 10    # Sync the 10 most recent runs"
echo "   Example: $0        # Sync the 5 most recent runs (default)"