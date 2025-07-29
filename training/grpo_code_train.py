"""GRPO training for Qwen code generation agent."""

import os
import sys
import yaml
import argparse
import torch
import numpy as np
import wandb
from datetime import datetime
from typing import Dict, Any, List

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from training.wandb_logger import WandBLogger
from environment.code_game_wrapper import CodeGameWrapper
from agents.qwen_code_agent import StaticQwenOpponent
from evaluation.evalplus_runner import EvalPlusRunner


class GRPOCodeTrainer:
    """GRPO trainer for code generation game."""
    
    def __init__(self, config_path: str):
        """Initialize GRPO code trainer."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"Loaded config from {config_path}")
        self.setup_logging()
        self.setup_environment()
        self.setup_models()
        self.setup_dataset()
        
    def setup_logging(self):
        """Initialize Weights & Biases logging."""
        wandb.login()
        
        self.logger = WandBLogger(
            project=self.config['logging']['project'] + "_code_game",
            entity=self.config['logging'].get('entity'),
            name=f"grpo-code-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=self.config,
        )
    
    def setup_environment(self):
        """Setup code generation game environment."""
        # Create static opponent
        opponent_config = self.config.get('static_opponent', {})
        self.static_opponent = StaticQwenOpponent(
            model_name=opponent_config.get('model_name', 'HuggingFaceTB/SmolLM-135M-Instruct'),
            device=opponent_config.get('device', 'auto')
        )
        
        # Create environment
        self.env = CodeGameWrapper(
            static_opponent_model=self.static_opponent,
            reward_shaper=None  # Can add custom reward shaping later
        )
        
        print("Code game environment setup completed")
    
    def setup_models(self):
        """Setup Qwen model for training."""
        qwen_config = self.config['qwen']
        lora_config = self.config['lora']
        
        # Load base model
        model_id = qwen_config['model_name']
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup LoRA
        lora_config_obj = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules="all-linear",
        )
        
        self.model = get_peft_model(self.model, lora_config_obj)
        print(f"LoRA parameters: {self.model.print_trainable_parameters()}")
        
        print("Model setup completed")
    
    def setup_dataset(self):
        """Setup dataset for GRPO training."""
        # Create dataset with the code generation prompt
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
        
        # Create dataset with repeated prompts
        num_samples = self.config['training'].get('num_samples', 1000)
        self.dataset = Dataset.from_dict({
            "prompt": [prompt] * num_samples
        })
        
        print(f"Created dataset with {len(self.dataset)} samples")
    
    def reward_function(self, completions: List[str], **kwargs) -> List[float]:
        """Reward function for GRPO training."""
        rewards = []
        
        for completion in completions:
            if not isinstance(completion, str):
                rewards.append(-1.0)
                continue
            
            try:
                # Create game state
                state = self.env.reset()
                
                # Take step with completion as action
                next_state, reward, done, info = self.env.step(completion)
                
                # Log episode info for debugging
                if len(rewards) < 5:  # Only log first few episodes to avoid spam
                    print("=" * 50)
                    print(f"Code: {info.get('code', 'N/A')}")
                    print(f"Expected: {info.get('expected_output', 'N/A')}")
                    print(f"True: {info.get('true_output', 'N/A')}")
                    print(f"Opponent: {info.get('opponent_prediction', 'N/A')}")
                    print(f"Reward: {reward}")
                    print(f"Completion: {completion[:200]}...")
                    print("=" * 50)
                
                rewards.append(float(reward))
                
            except Exception as e:
                print(f"Error in reward function: {e}")
                rewards.append(-5.0)  # Penalty for errors
        
        return rewards
    
    def setup_trainer(self):
        """Setup GRPO trainer."""
        training_config = self.config['training']
        
        # Training arguments
        training_args = GRPOConfig(
            output_dir=self.config['checkpoints']['save_dir'] + "_grpo",
            learning_rate=float(training_config['learning_rate']),
            per_device_train_batch_size=training_config['batch_size'],
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 2),
            max_prompt_length=training_config.get('max_prompt_length', 512),
            max_completion_length=training_config.get('max_completion_length', 200),
            num_generations=training_config.get('num_generations', 8),
            optim="adamw_8bit",
            num_train_epochs=training_config['num_epochs'],
            bf16=False,  # Disable BF16 for compatibility
            report_to=["wandb"],
            remove_unused_columns=False,
            logging_steps=1,
        )
        
        # Create trainer
        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=[self.reward_function],
            args=training_args,
            train_dataset=self.dataset,
        )
        
        print("GRPO trainer setup completed")
    
    def evaluate_coding_performance(self, checkpoint_path: str = None) -> Dict[str, Any]:
        """Evaluate model on coding benchmarks."""
        try:
            eval_config = self.config.get('evaluation', {})
            
            print("Running coding benchmark evaluation...")
            
            # Use current model or specified checkpoint
            model_path = checkpoint_path or self.config['qwen']['model_name']
            
            runner = EvalPlusRunner(
                model_path=model_path,
                lora_path=checkpoint_path,
                device=self.config['qwen'].get('device', 'auto'),
                max_new_tokens=512,
                temperature=0.2,
                num_samples=eval_config.get('num_coding_samples', 1),
                quick_eval=eval_config.get('quick_eval', True),
                quick_eval_size=eval_config.get('quick_eval_size', 10),
            )
            
            results = runner.run_evaluation()
            print(f"Coding evaluation results: {results}")
            
            return results
            
        except Exception as e:
            print(f"Coding evaluation failed: {e}")
            return {
                'humaneval_plus': {'pass@1': 0.0, 'pass@10': 0.0},
                'mbpp_plus': {'pass@1': 0.0, 'pass@10': 0.0},
                'eval_mode': 'Failed',
                'total_eval_time_seconds': 0,
            }
    
    def train(self):
        """Run GRPO training."""
        print("Starting GRPO code generation training...")
        
        # Setup trainer
        self.setup_trainer()
        
        # Initialize wandb run
        wandb.init(project=self.config['logging']['project'] + "_code_game")
        
        # Run initial evaluation
        if not self.config.get('evaluation', {}).get('skip_initial_eval', False):
            initial_results = self.evaluate_coding_performance()
            self.logger.log_coding_evaluation(initial_results, 0)
        
        # Train model
        self.trainer.train()
        
        # Final evaluation
        final_checkpoint = os.path.join(
            self.config['checkpoints']['save_dir'] + "_grpo",
            "final_checkpoint"
        )
        
        if os.path.exists(final_checkpoint):
            final_results = self.evaluate_coding_performance(final_checkpoint)
            self.logger.log_coding_evaluation(final_results, -1)  # -1 indicates final
        
        print("GRPO training completed!")
        self.logger.finish()


def create_config():
    """Create a default configuration for GRPO code training."""
    config = {
        'logging': {
            'project': 'qwen-code-game',
            'entity': None
        },
        'qwen': {
            'model_name': 'Qwen/Qwen2.5-3B',
            'device': 'auto',
            'temperature': 0.7,
            'max_new_tokens': 200
        },
        'static_opponent': {
            'model_name': 'HuggingFaceTB/SmolLM-135M-Instruct',
            'device': 'auto'
        },
        'lora': {
            'r': 16,
            'lora_alpha': 32,
            'target_modules': 'all-linear'
        },
        'training': {
            'num_samples': 1000,
            'num_epochs': 1,
            'batch_size': 8,
            'learning_rate': 2e-5,
            'gradient_accumulation_steps': 2,
            'max_prompt_length': 512,
            'max_completion_length': 200,
            'num_generations': 8
        },
        'checkpoints': {
            'save_dir': 'checkpoints/grpo_code',
            'save_interval': 1
        },
        'evaluation': {
            'skip_initial_eval': False,
            'quick_eval': True,
            'quick_eval_size': 10,
            'num_coding_samples': 1
        }
    }
    
    return config


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Qwen Code Generation GRPO agent")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/grpo_code.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default config file and exit"
    )
    
    args = parser.parse_args()
    
    if args.create_config:
        # Create default config
        config = create_config()
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        
        with open(args.config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"Created default config at: {args.config}")
        return
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Create one with: python training/grpo_code_train.py --create-config")
        return
    
    trainer = GRPOCodeTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()