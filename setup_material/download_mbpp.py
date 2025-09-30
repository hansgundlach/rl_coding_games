#!/usr/bin/env python3
"""Simple MBPP dataset downloader."""

import os
import json
import urllib.request

def download_mbpp():
    """Download MBPP dataset."""
    os.makedirs("evaluation/datasets", exist_ok=True)
    
    url = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json"
    path = "evaluation/datasets/sanitized-mbpp.json"
    
    try:
        print("Downloading MBPP dataset...")
        urllib.request.urlretrieve(url, path)
        
        # Verify
        with open(path, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Downloaded {len(data)} MBPP problems to {path}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Manual download: Place sanitized-mbpp.json in evaluation/datasets/")
        return False

if __name__ == "__main__":
    download_mbpp()