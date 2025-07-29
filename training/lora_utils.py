"""LoRA utilities for model fine-tuning."""

import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Optional, Dict, Any


def setup_lora_model(
    model_name: str,
    lora_config: dict,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
):
    """Setup model with LoRA configuration."""
    print(f"Loading base model: {model_name}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    # Create LoRA configuration
    lora_config_obj = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        target_modules=lora_config.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        bias=lora_config.get("bias", "none"),
    )

    # Apply LoRA to model
    peft_model = get_peft_model(model, lora_config_obj)
    peft_model.print_trainable_parameters()

    return peft_model


def merge_and_save_model(peft_model, output_dir: str, tokenizer=None) -> None:
    """Merge LoRA weights and save full model."""
    print(f"Merging LoRA weights and saving to {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Merge and unload the model
    merged_model = peft_model.merge_and_unload()

    # Save the merged model
    merged_model.save_pretrained(output_dir)

    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    print(f"Model saved successfully to {output_dir}")


def get_trainable_params_info(model) -> dict:
    """Get information about trainable parameters."""
    trainable_params = 0
    all_params = 0

    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_percentage = (
        (trainable_params / all_params) * 100 if all_params > 0 else 0
    )

    return {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": trainable_percentage,
    }


def load_lora_model(
    base_model_name: str,
    lora_checkpoint_path: str,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
):
    """Load a model with LoRA weights from checkpoint."""
    print(f"Loading base model: {base_model_name}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    # Load LoRA weights
    print(f"Loading LoRA weights from: {lora_checkpoint_path}")
    peft_model = PeftModel.from_pretrained(
        model,
        lora_checkpoint_path,
        torch_dtype=torch_dtype,
    )

    return peft_model


def save_lora_checkpoint(peft_model, checkpoint_dir: str) -> None:
    """Save LoRA adapter weights as checkpoint."""
    print(f"Saving LoRA checkpoint to {checkpoint_dir}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save only the LoRA adapter weights
    peft_model.save_pretrained(checkpoint_dir)

    print(f"LoRA checkpoint saved to {checkpoint_dir}")


def prepare_model_for_training(model) -> None:
    """Prepare model for training by enabling gradient computation for LoRA parameters."""
    # Enable gradient computation for trainable parameters
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradients for LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    # Ensure model is in training mode
    model.train()

    print("Model prepared for LoRA training")
