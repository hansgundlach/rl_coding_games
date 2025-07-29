"""Qwen agent for code generation game."""

import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class QwenCodeAgent:
    """Qwen agent for generating code that fools opponents."""
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-3B",
        device: str = "auto",
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        lora_path: Optional[str] = None
    ):
        """Initialize Qwen code generation agent.
        
        Args:
            model_name: Name of the Qwen model to use
            device: Device to run model on
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            lora_path: Path to LoRA checkpoint if available
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device if device != "auto" else "auto"
        )
        
        # Load LoRA weights if provided
        if lora_path:
            self._load_lora_weights(lora_path)
            
        # Create generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device if device != "auto" else "auto"
        )
        
        print(f"Loaded Qwen code agent: {model_name}")
        
    def _load_lora_weights(self, lora_path: str):
        """Load LoRA weights into the model."""
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"Loaded LoRA weights from: {lora_path}")
        except Exception as e:
            print(f"Warning: Failed to load LoRA weights from {lora_path}: {e}")
    
    def act(self, state: Dict[str, Any], config: Dict[str, Any] = None) -> str:
        """Generate code response for the given prompt.
        
        Args:
            state: Dictionary containing the game prompt
            config: Optional configuration parameters
            
        Returns:
            Generated code response as string
        """
        prompt = state.get('prompt', '')
        if not prompt:
            return self._generate_fallback_response()
            
        try:
            # Generate response
            response = self.generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )[0]['generated_text']
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._generate_fallback_response()
    
    def _generate_fallback_response(self) -> str:
        """Generate a simple fallback response if generation fails."""
        return """```python
def simple():
    return 42
print(simple())
```
```output
42
```"""
    
    def get_action_probabilities(self, state: Dict[str, Any]) -> torch.Tensor:
        """Get action probabilities for training (simplified version)."""
        # For text generation, this would need proper logit computation
        # Simplified for PPO training compatibility
        return torch.ones(1)  # Placeholder
    
    def compute_log_prob(self, state: Dict[str, Any], action: str) -> torch.Tensor:
        """Compute log probability of generated action."""
        prompt = state.get('prompt', '')
        full_text = prompt + action
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                full_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            if next(self.model.parameters()).device != torch.device("cpu"):
                inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Compute log probabilities (simplified)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Return mean log prob (simplified approximation)
            return log_probs.mean()
            
        except Exception as e:
            print(f"Error computing log probability: {e}")
            return torch.tensor(0.0)


class StaticQwenOpponent:
    """Static Qwen opponent that tries to guess code outputs."""
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-135M-Instruct",
        device: str = "auto"
    ):
        """Initialize static opponent.
        
        Args:
            model_name: Name of the model to use as opponent
            device: Device to run on
        """
        self.model_name = model_name
        self.device = device
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Move to device
        if torch.cuda.is_available() and device != "cpu":
            self.model = self.model.to("cuda")
        
        # Create pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        print(f"Loaded static opponent: {model_name}")
    
    def __call__(self, prompt: str, **kwargs) -> list:
        """Generate prediction for the given code."""
        return self.generator(prompt, **kwargs)