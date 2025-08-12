#!/usr/bin/env python3
"""
Test script to verify quantization functionality.
"""

import os
import sys
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add project path to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_quantization_config():
    """Test that quantization configuration is valid."""
    print("🧪 Testing quantization configuration...")

    # Test config
    config = {
        "model": {
            "id": "Qwen/Qwen3-1.7B",
            "cache_dir": "./model_cache",
            "quantization": {
                "enabled": True,
                "load_in_4bit": True,
                "load_in_8bit": False,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            },
        }
    }

    # Test BitsAndBytesConfig creation
    quantization_settings = config["model"]["quantization"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quantization_settings.get("load_in_4bit", False),
        load_in_8bit=quantization_settings.get("load_in_8bit", False),
        bnb_4bit_compute_dtype=getattr(
            torch, quantization_settings.get("bnb_4bit_compute_dtype", "float16")
        ),
        bnb_4bit_use_double_quant=quantization_settings.get(
            "bnb_4bit_use_double_quant", True
        ),
        bnb_4bit_quant_type=quantization_settings.get("bnb_4bit_quant_type", "nf4"),
    )

    print(f"✅ BitsAndBytesConfig created successfully")
    print(f"   load_in_4bit: {quantization_config.load_in_4bit}")
    print(f"   load_in_8bit: {quantization_config.load_in_8bit}")
    print(f"   bnb_4bit_compute_dtype: {quantization_config.bnb_4bit_compute_dtype}")
    print(
        f"   bnb_4bit_use_double_quant: {quantization_config.bnb_4bit_use_double_quant}"
    )
    print(f"   bnb_4bit_quant_type: {quantization_config.bnb_4bit_quant_type}")

    return True


def test_model_loading_with_quantization():
    """Test loading a small model with quantization (if GPU available)."""
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping model loading test")
        return True

    print("🧪 Testing model loading with quantization...")

    try:
        # Use a smaller model for testing
        model_id = "microsoft/DialoGPT-small"  # Much smaller than Qwen for testing

        # Create quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype="auto",
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"✅ Model loaded successfully with quantization")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test a simple forward pass
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        print(f"✅ Forward pass successful")
        print(f"   Output shape: {outputs.logits.shape}")

        return True

    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False


def test_config_file_loading():
    """Test loading the quantized config file."""
    print("🧪 Testing quantized config file loading...")

    config_path = "configs/grpo_prisoners_dilemma_quantized.yaml"

    if not os.path.exists(config_path):
        print(f"⚠️ Config file not found: {config_path}")
        return True  # Not a failure, just missing file

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check quantization settings
        quantization = config["model"].get("quantization", {})

        print(f"✅ Config file loaded successfully")
        print(f"   Quantization enabled: {quantization.get('enabled', False)}")
        print(f"   4-bit quantization: {quantization.get('load_in_4bit', False)}")
        print(f"   8-bit quantization: {quantization.get('load_in_8bit', False)}")

        return True

    except Exception as e:
        print(f"❌ Config file loading failed: {e}")
        return False


def main():
    """Run all quantization tests."""
    print("🚀 Starting quantization tests...")

    tests = [
        test_quantization_config,
        test_config_file_loading,
        test_model_loading_with_quantization,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            print()

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All quantization tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
