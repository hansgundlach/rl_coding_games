#!/bin/bash

# Script to sync the 5 most recent offline wandb runs
# All offline runs should be in the wandb/ folder

echo "🔄 Finding and syncing the 5 most recent offline wandb runs..."
echo

# Check if wandb directory exists
if [ ! -d "wandb" ]; then
    echo "❌ Error: wandb directory not found in current directory"
    exit 1
fi

# Find all run directories (excluding latest-run which is typically a symlink)
# Sort by timestamp in directory name (newest first) and take the first 5
recent_runs=$(find wandb -maxdepth 1 -type d -name "run-*" | grep -v "latest-run" | sort -r | head -5)

# Check if any runs were found
if [ -z "$recent_runs" ]; then
    echo "❌ No offline wandb runs found in wandb/ directory"
    exit 1
fi

echo "📋 Found the following recent runs to sync:"
echo "$recent_runs" | nl -w2 -s'. '
echo

# Sync each run
count=0
for run_dir in $recent_runs; do
    count=$((count + 1))
    echo "🚀 [$count/5] Syncing: $run_dir"
    
    if wandb sync "$run_dir"; then
        echo "✅ Successfully synced: $run_dir"
    else
        echo "❌ Failed to sync: $run_dir"
    fi
    echo
done

echo "🎉 Finished syncing offline wandb runs!"