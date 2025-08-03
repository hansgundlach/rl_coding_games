#!/bin/bash

# Script to sync the 5 most recent offline wandb runs
# Offline runs are stored in wandb/ directory in a special format

echo "🔄 Finding and syncing the 5 most recent offline wandb runs..."
echo

# Check if wandb directory exists
if [ ! -d "wandb" ]; then
    echo "❌ Error: wandb directory not found in current directory"
    exit 1
fi

# For offline runs, we need to sync from the wandb directory
# First, let's find the most recent offline runs by checking for .wandb files
echo "🔍 Scanning for offline wandb runs..."

# Find all .wandb files (these indicate offline runs) and sort by modification time
recent_wandb_files=$(find wandb -name "*.wandb" -type f -printf '%T@ %p\n' | sort -rn | head -5 | cut -d' ' -f2-)

# Check if any offline runs were found
if [ -z "$recent_wandb_files" ]; then
    echo "❌ No offline wandb runs found in wandb/ directory"
    exit 1
fi

echo "📋 Found the following recent offline runs:"
count=0
for wandb_file in $recent_wandb_files; do
    count=$((count + 1))
    run_dir=$(dirname "$wandb_file")
    echo "$count. $run_dir"
done
echo

# Method 1: Sync all offline runs at once (recommended)
echo "🚀 Method 1: Syncing all recent offline runs at once..."
if wandb sync wandb/; then
    echo "✅ Successfully synced all offline runs from wandb/ directory"
else
    echo "❌ Batch sync failed, trying individual sync..."
    
    # Method 2: Sync individual runs if batch fails
    echo "🚀 Method 2: Syncing individual runs..."
    count=0
    for wandb_file in $recent_wandb_files; do
        count=$((count + 1))
        run_dir=$(dirname "$wandb_file")
        echo "🚀 [$count/5] Syncing: $run_dir"
        
        # For individual offline runs, sync from the run directory
        if (cd "$run_dir" && wandb sync .); then
            echo "✅ Successfully synced: $run_dir"
        else
            echo "❌ Failed to sync: $run_dir"
        fi
        echo
    done
fi

echo "🎉 Finished syncing offline wandb runs!"