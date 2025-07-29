#!/usr/bin/env python3
"""Quick test of the GRPO code generation system."""

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import GRPOConfig, GRPOTrainer
import re
import io
import contextlib
import wandb
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.env_loader import get_api_key

# Initialize wandb with API key from environment
wandb_key = get_api_key('wandb', required=False)
if wandb_key:
    wandb.login(key=wandb_key)
    print("✓ Logged into W&B using environment variable")
else:
    print("⚠️ No W&B API key found, continuing without logging")

def run_and_capture(code: str) -> str:
    """Executes code and captures its stdout output."""
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {})
    except Exception as e:
        return f"Execution error: {e}"
    return buffer.getvalue().strip()

# Define the formatted prompt
prompt = (
    "Write a Python program that is difficult for another model trained on SmollM-135M-Instruct data to predict. "
    "The program should return a singular integer value. "
    "Then, show only the exact output of running your program.\n\n"

    "Format your response exactly like these examples:\n\n"

    "```python\n"
    "def tricky():\n"
    "    return int('0b1011', 2)\n"
    "print(tricky())\n"
    "```\n"
    "```output\n"
    "11\n"
    "```\n\n"

    "```python\n"
    "def f():\n"
    "    return sum([i % 3 for i in range(10)])\n"
    "print(f())\n"
    "```\n"
    "```output\n"
    "10\n"
    "```\n\n"

    "Now you try:\n"
)

# Create a SMALL dataset for quick testing
dataset = Dataset.from_dict({
    "prompt": [prompt] * 50  # Only 50 samples for quick test
})

print(f"Created test dataset: {dataset}")

# Load the main model (trainable)
model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
print(f"Loading trainable model: {model_id}")

model1 = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer1 = AutoTokenizer.from_pretrained(model_id)

# Load LoRA (smaller for testing)
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,  # Smaller rank for faster testing
    lora_alpha=16,
    target_modules="all-linear",
)
model1 = get_peft_model(model1, lora_config)
print(model1.print_trainable_parameters())

# Load static opponent model
model_name2 = "HuggingFaceTB/SmolLM-135M-Instruct"
print(f"Loading static opponent: {model_name2}")

tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
model2 = AutoModelForCausalLM.from_pretrained(model_name2, torch_dtype="auto")
model2 = model2.to("cuda" if torch.cuda.is_available() else "cpu")

# Create text generation pipeline for opponent
generator2 = pipeline("text-generation", model=model2, tokenizer=tokenizer2)

def reward_function(completions, **kwargs):
    """Reward function for GRPO training."""
    print(f"\n🎯 REWARD FUNCTION CALLED with {len(completions)} completions")
    rewards = []
    
    for i, comp in enumerate(completions):
        print(f"\n--- Processing completion {i+1}/{len(completions)} ---")
        
        if not isinstance(comp, str):
            rewards.append(-1)
            print(f"❌ Invalid completion type: {type(comp)}")
            continue
        
        # Extract code
        code = re.search(r"```python\s*\n(.*?)```", comp, re.DOTALL)
        if code:
            code = code.group(1).strip()
        else:
            code = ""
        
        expected_output = re.search(r"```output\s*(.*?)```", comp, re.DOTALL)
        if expected_output:
            expected_output = expected_output.group(1).strip()
        else:
            expected_output = ""
        
        print(f"🐍 Code: {code[:100]}...")
        print(f"📝 Expected: {expected_output}")
        
        # Get opponent prediction
        prompt2 = (
            "Examine this code and predict the integer output.\n"
            f"{code}\n\n"
            "Do not include any text, markdown, or explanation, just the number."
        )
        
        try:
            model_pred = generator2(
                prompt2, 
                max_new_tokens=50,  # Shorter for speed
                do_sample=True, 
                temperature=0.7
            )[0]['generated_text']
            
            # Extract first number
            model_pred = re.search(r"\b(-?\d+)\b", model_pred)
            model_pred = model_pred.group(1) if model_pred else ""
            try:
                model_pred = int(model_pred)
            except:
                model_pred = "ERROR"
        except Exception as e:
            model_pred = f"ERROR: {e}"
        
        # Get true output
        true_output = run_and_capture(code)
        
        print(f"🤖 Opponent predicted: {model_pred}")
        print(f"✅ True output: {true_output}")
        
        # Calculate reward
        try:
            if str(model_pred) == str(true_output).strip():
                reward = -1  # Opponent was right
                print("💔 Opponent guessed correctly - penalty")
            else:
                reward = 1   # Opponent was wrong
                print("🎉 Opponent fooled - reward!")
        except:
            reward = -1
            print("⚠️ Error in comparison - penalty")
            
        rewards.append(reward)
        
        # Only show first few in detail
        if i >= 2:
            print(f"... (showing first 3 completions only)")
            break
    
    print(f"\n📊 Rewards: {rewards}")
    return rewards

# Quick training arguments
print("Setting up quick GRPO training...")
training_args = GRPOConfig(
    output_dir="checkpoints/grpo_quick_test",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Must be divisible by num_generations
    gradient_accumulation_steps=1,
    max_prompt_length=512,
    max_completion_length=100,  # Shorter completions
    num_generations=4,  # num_generations must divide batch_size
    optim="adamw_8bit",
    num_train_epochs=1,
    max_steps=2,  # Only 2 steps for testing
    bf16=False,
    report_to=["wandb"],
    remove_unused_columns=False,
    logging_steps=1,
)

# Create trainer
trainer = GRPOTrainer(
    model=model1,
    reward_funcs=[reward_function],
    args=training_args,
    train_dataset=dataset,
)

print("🚀 Starting quick GRPO test...")

# Initialize wandb run
wandb.init(project="qwen-code-game-grpo-test")

# Train model (just 2 steps)
trainer.train()

print("✅ Quick test completed!")
wandb.finish()