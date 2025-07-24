#!/usr/bin/env python3
"""
Quick setup verification script for Qwen ConnectX RL Training
Run this before starting training to ensure everything is properly configured.
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (need 3.10+)")
        return False

def check_gpu():
    """Check GPU availability and memory."""
    print("🎮 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            total_memory = 0
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1e9
                total_memory += memory_gb
                print(f"   ✅ GPU {i}: {props.name} ({memory_gb:.1f} GB)")
            
            if total_memory >= 8:
                print(f"   ✅ Total GPU memory: {total_memory:.1f} GB (sufficient)")
                return True
            else:
                print(f"   ⚠️  Total GPU memory: {total_memory:.1f} GB (may be insufficient)")
                return False
        else:
            print("   ⚠️  No CUDA GPUs detected - will use CPU (very slow)")
            return False
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✅ {package_name} ({version})")
        return True
    except ImportError:
        print(f"   ❌ {package_name} - not installed")
        return False

def check_required_packages():
    """Check all required packages."""
    print("📦 Checking required packages...")
    
    packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'), 
        ('peft', 'peft'),
        ('accelerate', 'accelerate'),
        ('wandb', 'wandb'),
        ('evalplus', 'evalplus'),
        ('kaggle-environments', 'kaggle_environments'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('pyyaml', 'yaml'),
    ]
    
    all_good = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_good = False
    
    return all_good

def check_huggingface_access():
    """Check Hugging Face authentication."""
    print("🤗 Checking Hugging Face access...")
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"   ✅ Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"   ⚠️  Not logged in to Hugging Face: {e}")
        print("      Run: huggingface-cli login")
        return False

def check_wandb_access():
    """Check Weights & Biases authentication."""
    print("📊 Checking Weights & Biases access...")
    try:
        import wandb
        api_key = wandb.api.api_key
        if api_key:
            print("   ✅ W&B API key configured")
            return True
        else:
            print("   ⚠️  W&B not configured (optional)")
            return False
    except Exception as e:
        print(f"   ⚠️  W&B check failed: {e}")
        return False

def check_project_structure():
    """Check project files and directories."""
    print("📁 Checking project structure...")
    
    required_files = [
        'training/qwen_ppo_train.py',
        'configs/qwen_ppo.yaml',
        'agents/qwen_policy.py',
        'environment/connectx_wrapper.py',
        'evaluation/evalplus_runner.py',
        'requirements.txt',
    ]
    
    required_dirs = [
        'checkpoints',
        'training',
        'configs', 
        'agents',
        'environment',
        'evaluation',
    ]
    
    all_good = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - missing")
            all_good = False
    
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ❌ {dir_path}/ - missing")
            all_good = False
    
    return all_good

def check_disk_space():
    """Check available disk space."""
    print("💾 Checking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        free_gb = free // (1024**3)
        
        if free_gb >= 20:
            print(f"   ✅ {free_gb} GB free space available")
            return True
        else:
            print(f"   ⚠️  Only {free_gb} GB free space (need 20+ GB)")
            return False
    except Exception as e:
        print(f"   ⚠️  Could not check disk space: {e}")
        return False

def check_config_files():
    """Check configuration files are valid."""
    print("⚙️  Checking configuration files...")
    try:
        import yaml
        
        config_files = ['configs/qwen_ppo.yaml', 'configs/ppo.yaml']
        all_good = True
        
        for config_file in config_files:
            if Path(config_file).exists():
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    print(f"   ✅ {config_file} - valid YAML")
                except Exception as e:
                    print(f"   ❌ {config_file} - invalid YAML: {e}")
                    all_good = False
            else:
                print(f"   ❌ {config_file} - missing")
                all_good = False
        
        return all_good
    except ImportError:
        print("   ❌ PyYAML not installed")
        return False

def main():
    """Run all setup checks."""
    print("=" * 50)
    print("🔍 Qwen ConnectX RL Setup Verification")
    print("=" * 50)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("GPU Support", check_gpu),
        ("Required Packages", check_required_packages),
        ("Project Structure", check_project_structure),
        ("Configuration Files", check_config_files),
        ("Disk Space", check_disk_space),
        ("Hugging Face Access", check_huggingface_access),
        ("Weights & Biases", check_wandb_access),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"   ❌ Error during {check_name}: {e}")
            results.append((check_name, False))
        print()
    
    # Summary
    print("=" * 50)
    print("📋 Setup Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:.<30} {status}")
        if result:
            passed += 1
    
    print()
    print(f"Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print()
        print("🎉 All checks passed! You're ready to start training:")
        print("   python training/qwen_ppo_train.py --config configs/qwen_ppo.yaml")
    elif passed >= total - 2:  # Allow some optional checks to fail
        print()
        print("⚠️  Most checks passed. Training should work but may have issues.")
        print("   Consider fixing failed checks for optimal experience.")
    else:
        print()
        print("❌ Multiple checks failed. Please fix issues before training.")
        print("   See COMPLETE_SETUP_GUIDE.md for detailed setup instructions.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())