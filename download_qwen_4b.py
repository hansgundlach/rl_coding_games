#!/usr/bin/env python3
"""
Download Qwen3-4B model and cache it locally
"""

import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers


def download_qwen_4b():
    """Download and cache Qwen3-4B model."""

    # Check transformers version for Qwen3 compatibility
    transformers_version = transformers.__version__
    print(f"📋 Transformers version: {transformers_version}")
    
    # Qwen3 requires transformers>=4.51.0
    from packaging import version
    if version.parse(transformers_version) < version.parse("4.51.0"):
        print(f"❌ Error: Qwen3-4B requires transformers>=4.51.0, but you have {transformers_version}")
        print("💡 Please upgrade transformers: pip install --upgrade transformers")
        sys.exit(1)
    
    # Model configuration
    model_id = "Qwen/Qwen3-4B"
    cache_dir = "./model_cache"
    
    print(f"🚀 Starting download of {model_id}")
    print(f"📁 Cache directory: {cache_dir}")

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    try:
        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir, trust_remote_code=True
        )
        print("✅ Tokenizer downloaded successfully")

        print("📥 Downloading model (this may take several minutes for 4B model)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=None,  # Don't load to GPU, just cache
        )
        print("✅ Model downloaded successfully")

        # Verify the download
        print(f"📊 Model parameters: {model.num_parameters():,}")
        print(f"📊 Tokenizer vocab size: {tokenizer.vocab_size}")

        # Check cache directory
        cached_model_path = os.path.join(
            cache_dir, f"models--{model_id.replace('/', '--')}"
        )
        if os.path.exists(cached_model_path):
            print(f"✅ Model successfully cached at: {cached_model_path}")

            # List snapshots
            snapshots_dir = os.path.join(cached_model_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                print(f"📋 Available snapshots: {snapshots}")

        print(f"🎉 {model_id} is now ready for use!")

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("💡 Make sure you have internet connection and sufficient disk space")
        print("💡 For 4B model, you'll need approximately 8-16GB of free space")
        sys.exit(1)


if __name__ == "__main__":
    download_qwen_4b()
