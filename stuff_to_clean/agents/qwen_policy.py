"""Qwen-3-4B based policy agent for Connect-X."""

import numpy as np
from typing import Any, Dict, Optional
from .base_agent import BaseAgent

# Lazy imports to avoid heavy dependencies during testing
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from utils.env_loader import get_api_key


class QwenPolicyAgent(BaseAgent):
    """Connect-X agent using Qwen-3-4B with LoRA fine-tuning."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B",
        lora_checkpoint: Optional[str] = None,
        device: str = "auto",
        max_new_tokens: int = 10,
        temperature: float = 0.7,
    ):
        """Initialize Qwen policy agent."""
        super().__init__(name="QwenPolicyAgent")
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Real model initialization
        self.tokenizer = None
        self.model = None
        self.lora_checkpoint = lora_checkpoint
        self._initialized = False

        # Get API key if using hosted models
        if "api" in model_name.lower():
            self.api_key = get_api_key("anthropic")  # or 'openai'

        # Initialize model and tokenizer
        self._lazy_init()

    def _lazy_init(self):
        """Initialize model components with proper error handling."""
        if not self._initialized and TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading Qwen model: {self.model_name}")

                # Check available memory
                if torch.cuda.is_available():
                    available_memory = torch.cuda.get_device_properties(0).total_memory
                    print(f"Available GPU memory: {available_memory / 1e9:.1f} GB")

                    if available_memory < 8e9:  # Less than 8GB
                        print("Warning: Limited GPU memory, may cause OOM errors")

                # Load tokenizer with error handling
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        cache_dir="./model_cache",
                    )
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                except Exception as e:
                    raise RuntimeError(f"Failed to load tokenizer: {e}")

                # Load model with memory optimization
                try:
                    model_kwargs = {
                        "torch_dtype": torch.float16,
                        "trust_remote_code": True,
                        "cache_dir": "./model_cache",
                        "low_cpu_mem_usage": True,
                        "local_files_only": False,  # Added
                    }

                    # Set device mapping based on available resources
                    if self.device == "auto":
                        if torch.cuda.is_available():
                            model_kwargs["device_map"] = "auto"
                        else:
                            model_kwargs["device_map"] = {"": "cpu"}
                    else:
                        model_kwargs["device_map"] = {"": self.device}

                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name, **model_kwargs
                    )

                except Exception as e:
                    raise RuntimeError(f"Failed to load model: {e}")

                # Load LoRA checkpoint if provided
                if self.lora_checkpoint:
                    self._load_lora_checkpoint()

                self.model.eval()
                self._initialized = True
                print("Qwen model loaded successfully")

                # Log memory usage
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    print(f"GPU memory used after loading: {memory_used:.1f} GB")

            except Exception as e:
                print(f"Failed to load Qwen model: {e}")
                print("Falling back to heuristic agent")
                self._initialized = False

                # Clean up partial initialization
                self._cleanup_resources()

    def act(self, observation: Dict[str, Any], configuration: Dict[str, Any]) -> int:
        """Generate action using Qwen model."""
        board = np.array(observation["board"]).reshape(
            configuration["rows"], configuration["columns"]
        )
        valid_actions = self.get_valid_actions(board)

        if not valid_actions:
            return 0  # Fallback

        # Use Qwen model if properly initialized
        if self._initialized and self.model is not None:
            try:
                return self._generate_qwen_action(board, configuration, valid_actions)
            except Exception as e:
                print(f"Qwen inference failed: {e}, falling back to heuristic")

        # Fallback to heuristic (center preference)
        center_col = configuration["columns"] // 2
        if center_col in valid_actions:
            return center_col

        return valid_actions[0]

    def _create_game_prompt(self, board: np.ndarray, player_mark: int) -> str:
        """Create prompt for Connect-X game state."""
        board_str = self._board_to_string(board)
        prompt = f"""Connect-X Game Analysis:
Board state:
{board_str}

You are player {player_mark}. Choose the best column (0-{board.shape[1]-1}).
Your move (column number):"""
        return prompt

    def _board_to_string(self, board: np.ndarray) -> str:
        """Convert board array to readable string representation."""
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = []
        for row in board:
            rows.append("".join(symbols[cell] for cell in row))
        return "\n".join(rows)

    def _generate_qwen_action(
        self, board: np.ndarray, configuration: Dict[str, Any], valid_actions: list
    ) -> int:
        """Generate action using Qwen model inference with error handling."""
        try:
            # Determine current player mark
            total_pieces = np.sum(board > 0)
            player_mark = 1 if total_pieces % 2 == 0 else 2

            # Create prompt
            prompt = self._create_game_prompt(board, player_mark)

            # Tokenize with error handling
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
            except Exception as e:
                raise RuntimeError(f"Tokenization failed: {e}")

            # Move inputs to device
            if hasattr(self.model, "device") and self.model.device != torch.device(
                "cpu"
            ):
                try:
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                except Exception as e:
                    raise RuntimeError(f"Failed to move inputs to device: {e}")

            # Generate response with timeout protection
            try:
                with torch.no_grad():
                    # Clear GPU cache before generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=min(self.max_new_tokens, 20),  # Limit tokens
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        early_stopping=True,
                    )
            except torch.cuda.OutOfMemoryError:
                # Handle OOM by clearing cache and retrying with minimal settings
                torch.cuda.empty_cache()
                print(
                    "Warning: GPU OOM during generation, retrying with reduced settings"
                )
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=5,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            except Exception as e:
                raise RuntimeError(f"Generation failed: {e}")

            # Decode response
            try:
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()
            except Exception as e:
                raise RuntimeError(f"Decoding failed: {e}")

            # Parse action from generated text
            action = self._parse_action_from_text(generated_text, valid_actions)
            return action

        except Exception as e:
            print(f"Qwen inference error: {e}")
            # Fallback to heuristic
            return self._heuristic_action(valid_actions)

    def _parse_action_from_text(self, text: str, valid_actions: list) -> int:
        """Parse action from Qwen generated text."""
        # Try to extract column number from the text
        import re

        # Look for numbers in the generated text
        numbers = re.findall(r"\b\d+\b", text)

        for num_str in numbers:
            try:
                action = int(num_str)
                if action in valid_actions:
                    return action
            except ValueError:
                continue

        # If no valid action found, return first valid action
        return valid_actions[0]

    def _load_lora_checkpoint(self):
        """Load LoRA checkpoint into the model."""
        try:
            from peft import PeftModel

            print(f"Loading LoRA checkpoint: {self.lora_checkpoint}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.lora_checkpoint,
                torch_dtype=torch.float16,
            )
            print("LoRA checkpoint loaded successfully")
        except Exception as e:
            print(f"Failed to load LoRA checkpoint: {e}")
            # Continue without LoRA

    def _heuristic_action(self, valid_actions: list) -> int:
        """Fallback heuristic action when Qwen inference fails."""
        # Simple center preference heuristic
        center_positions = [3, 2, 4, 1, 5, 0, 6]  # Center-out preference
        for pos in center_positions:
            if pos in valid_actions:
                return pos
        return valid_actions[0] if valid_actions else 0

    def _cleanup_resources(self):
        """Clean up model resources."""
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
                self.model = None

            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("Resources cleaned up")

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self._cleanup_resources()
