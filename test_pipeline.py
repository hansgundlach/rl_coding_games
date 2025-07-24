"""Test script to verify the complete Qwen RL pipeline."""

import os
import sys
import torch
import traceback
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
        
        import peft
        print(f"✓ PEFT {peft.__version__}")
        
        from agents.qwen_policy import QwenPolicyAgent
        print("✓ QwenPolicyAgent")
        
        from training.lora_utils import setup_lora_model
        print("✓ LoRA utilities")
        
        from evaluation.evalplus_runner import EvalPlusRunner
        print("✓ EvalPlus runner")
        
        from environment.connectx_wrapper import ConnectXWrapper
        print("✓ ConnectX environment")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_qwen_agent():
    """Test Qwen agent initialization and basic functionality."""
    print("\nTesting Qwen agent...")
    
    try:
        from agents.qwen_policy import QwenPolicyAgent
        
        # Test with fallback to heuristic if model loading fails
        agent = QwenPolicyAgent(
            model_name="Qwen/Qwen2.5-3B",
            device="cpu",  # Use CPU for testing
            max_new_tokens=5,
            temperature=0.7,
        )
        
        # Test action generation
        test_observation = {
            'board': [0] * 42  # Empty 6x7 board
        }
        test_config = {
            'rows': 6,
            'columns': 7
        }
        
        action = agent.act(test_observation, test_config)
        print(f"✓ Agent generated action: {action}")
        
        # Cleanup
        agent._cleanup_resources()
        
        return True
        
    except Exception as e:
        print(f"✗ Qwen agent test failed: {e}")
        traceback.print_exc()
        return False

def test_lora_setup():
    """Test LoRA model setup (without actually loading the model)."""
    print("\nTesting LoRA setup...")
    
    try:
        from training.lora_utils import get_trainable_params_info
        
        # Create a simple model for testing
        import torch.nn as nn
        test_model = nn.Linear(10, 1)
        
        info = get_trainable_params_info(test_model)
        print(f"✓ LoRA utils working - trainable params: {info['trainable_params']}")
        
        return True
        
    except Exception as e:
        print(f"✗ LoRA setup test failed: {e}")
        return False

def test_environment():
    """Test ConnectX environment."""
    print("\nTesting ConnectX environment...")
    
    try:
        from environment.connectx_wrapper import ConnectXWrapper
        
        env = ConnectXWrapper(
            opponent_agent="random",
            rows=6,
            columns=7,
        )
        
        # Test environment reset and step
        state = env.reset()
        print(f"✓ Environment reset - state shape: {state.shape}")
        
        # Test a step
        action = 3  # Middle column
        next_state, reward, done, info = env.step(action)
        print(f"✓ Environment step - reward: {reward}, done: {done}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_config_files():
    """Test that configuration files are valid."""
    print("\nTesting configuration files...")
    
    try:
        import yaml
        
        # Test regular PPO config
        with open("configs/ppo.yaml", 'r') as f:
            ppo_config = yaml.safe_load(f)
        print("✓ PPO config loaded")
        
        # Test Qwen PPO config
        with open("configs/qwen_ppo.yaml", 'r') as f:
            qwen_config = yaml.safe_load(f)
        print("✓ Qwen PPO config loaded")
        
        # Validate required fields
        required_sections = ['environment', 'qwen', 'lora', 'training', 'logging']
        for section in required_sections:
            if section not in qwen_config:
                raise ValueError(f"Missing required section: {section}")
        
        print("✓ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def check_gpu_availability():
    """Check GPU availability and memory."""
    print("\nChecking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ CUDA available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        return True
    else:
        print("ℹ No GPU available - will use CPU")
        return False

def main():
    """Run all tests."""
    print("=== Qwen ConnectX RL Pipeline Test ===\n")
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    test_results = []
    
    # Run tests
    test_results.append(("Imports", test_imports()))
    test_results.append(("GPU Check", check_gpu_availability()))
    test_results.append(("Configuration", test_config_files()))
    test_results.append(("Environment", test_environment()))
    test_results.append(("LoRA Setup", test_lora_setup()))
    test_results.append(("Qwen Agent", test_qwen_agent()))
    
    # Summary
    print("\n=== Test Summary ===")
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to use.")
        print("\nTo start training:")
        print("python training/qwen_ppo_train.py --config configs/qwen_ppo.yaml")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()