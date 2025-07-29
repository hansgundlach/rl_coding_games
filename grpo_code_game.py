#!/usr/bin/env python3
"""
GRPO Code Generation Game - Qwen vs Static Opponent
Based on the working RFT One Way example
"""

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
import datetime

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
            exec(code, {})  # empty global scope
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

# Create a dataset of 1000 identical prompts
dataset = Dataset.from_dict({
    "prompt": [prompt] * 1000
})

print(f"Created dataset: {dataset}")

# Load the main model (trainable)
model_id = "Qwen/Qwen2.5-1.5B"
print(f"Loading trainable model: {model_id}")

model1 = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer1 = AutoTokenizer.from_pretrained(model_id)

# Load LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)
model1 = get_peft_model(model1, lora_config)
print(model1.print_trainable_parameters())

# Load static opponent model
model_name2 = "Qwen/Qwen2.5-1.5B"
print(f"Loading static opponent: {model_name2}")

tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
model2 = AutoModelForCausalLM.from_pretrained(model_name2, torch_dtype="auto")

# Move to GPU if available
model2 = model2.to("cuda" if torch.cuda.is_available() else "cpu")

# Create text generation pipeline for opponent
generator2 = pipeline("text-generation", model=model2, tokenizer=tokenizer2)

def reward_function(completions, **kwargs):
    """Reward function for GRPO training."""
    rewards = []
    
    successful_predictions = 0 # Trainable model "wins" (opponent fails to predict correctly)
    failed_predictions = 0     # Trainable model "losses" (opponent predicts correctly)
    
    for i, comp in enumerate(completions):
        if not isinstance(comp, str):
            rewards.append(-1)  # invalid completion
            failed_predictions += 1
            continue
        
        # Extract code according to schema
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
        
        # Get opponent prediction
        prompt2 = (
            "Examine this code and predict the integer output.\n"
            f"{code}\n\n"
            "Do not include any text, markdown, or explanation, just the number."
        )
        
        model_pred = "" # Initialize model_pred to ensure it's always defined
        try:
            model_pred_raw = generator2(
                prompt2, 
                max_new_tokens=200, 
                do_sample=True, 
                temperature=0.7
            )[0]['generated_text']
            
            # Extract first number from prediction
            model_pred_search = re.search(r"\b(-?\d+)\b", model_pred_raw)
            model_pred = model_pred_search.group(1) if model_pred_search else ""
            try:
                model_pred = int(model_pred)
            except ValueError:
                model_pred = "ERROR: Conversion to integer failed"
        except Exception as e:
            model_pred = f"ERROR: {e}"
        
        # Get true output
        true_output = run_and_capture(code)
        
        # Debug output (show first few)
        if len(rewards) < 3:
            print("=" * 50)
            print(f'Model prediction: {model_pred}')
            print(f'Code: {code}')
            print(f'Expected output: {expected_output}')
            print(f'True output: {true_output}')
            print(f'Original completion: {comp[:200]}...')
            print("=" * 50)
        
        # Calculate reward: +1 if opponent is wrong, -1 if opponent is right
        reward = -1 # Default to error/loss
        try:
            # Convert both to string for robust comparison, as model_pred can be int or error string
            if str(model_pred) == str(true_output).strip():
                reward = -1  # Opponent guessed correctly
                failed_predictions += 1
            else:
                reward = 1   # Opponent guessed incorrectly (good for us)
                successful_predictions += 1
        except Exception as e:
            print(f"Error comparing outputs in reward_function: {e}")
            reward = -1  # Error case, treat as a loss for the trainable model
            failed_predictions += 1
        
        # Show progress for all completions
        print(f"Completion {i+1}/{len(completions)}: pred={model_pred}, true={true_output}, reward={reward}")
            
        rewards.append(reward)
    
    # Log metrics to wandb after processing the entire batch of completions
    if wandb.run: # Ensure wandb is initialized before logging
        avg_batch_reward = sum(rewards) / len(rewards) if rewards else 0
        
        wandb.log({
            "reward/avg_batch_reward": avg_batch_reward,
            "reward/successful_predictions_count": successful_predictions,
            "reward/failed_predictions_count": failed_predictions,
            "reward/total_completions_processed": len(completions)
        })
    
    return rewards

# Training arguments
print("Setting up GRPO training...")
training_args = GRPOConfig(
    output_dir="checkpoints/grpo_code",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_prompt_length=512,
    max_completion_length=200,
    num_generations=2,
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=False,  # Disabled for compatibility
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

print("Starting GRPO training...")

# Initialize wandb run
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
wandb.init(project=f"qwen-code-game-grpo-{timestamp}")

# Train model
trainer.train()

print("Training completed!")
wandb.finish()