#!/bin/bash

# Script to sync the 5 most recent offline wandb runs
# All offline runs should be in the wandb/ folder with prefix "offline-run-"

echo "🔄 Finding and syncing the 5 most recent offline wandb runs..."
echo

# Check if wandb directory exists
if [ ! -d "wandb" ]; then
    echo "❌ Error: wandb directory not found in current directory"
    exit 1
fi

# Change to wandb directory
cd wandb

# Find all offline-run directories, sort by modification time (newest first) and take the first 5
recent_runs=$(find . -maxdepth 1 -type d -name "offline-run-*" -printf '%T@ %p\n' | sort -rn | head -5 | cut -d' ' -f2-)

# Check if any runs were found
if [ -z "$recent_runs" ]; then
    echo "❌ No offline wandb runs found in wandb/ directory"
    echo "Looking for directories with pattern: offline-run-*"
    exit 1
fi

echo "📋 Found the following recent offline runs:"
echo "$recent_runs" | sed 's|^\./||' | nl -w2 -s'. '
echo

# Sync each run
count=0
for run_dir in $recent_runs; do
    count=$((count + 1))
    run_name=$(basename "$run_dir")
    echo "🚀 [$count/5] Syncing: $run_name"
    
    if wandb sync "$run_name"; then
        echo "✅ Successfully synced: $run_name"
    else
        echo "❌ Failed to sync: $run_name"
    fi
    echo
done

echo "🎉 Finished syncing offline wandb runs!"