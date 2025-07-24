"""EvalPlus benchmark runner for code generation evaluation."""

import os
import json
import tempfile
import subprocess
import logging
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvalPlusRunner:
    """Runner for HumanEval-Plus and MBPP-Plus benchmarks."""
    
    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        num_samples: int = 10,
    ):
        """Initialize EvalPlus runner."""
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.num_samples = num_samples
        
        logger.info(f"Initializing EvalPlus with model: {model_path}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            
            # Load LoRA if specified
            if self.lora_path:
                from peft import PeftModel
                logger.info(f"Loading LoRA weights from {self.lora_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_path,
                    torch_dtype=torch.float16,
                )
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite."""
        logger.info("Starting EvalPlus evaluation...")
        
        results = {}
        
        # Run HumanEval-Plus
        logger.info("Running HumanEval-Plus evaluation...")
        humaneval_results = self.evaluate_humaneval_plus()
        results['humaneval_plus'] = humaneval_results
        
        # Run MBPP-Plus
        logger.info("Running MBPP-Plus evaluation...")
        mbpp_results = self.evaluate_mbpp_plus()
        results['mbpp_plus'] = mbpp_results
        
        logger.info("Evaluation completed")
        return results
    
    def evaluate_humaneval_plus(self) -> Dict[str, float]:
        """Evaluate on HumanEval-Plus benchmark."""
        try:
            # Install evalplus if not available
            self._ensure_evalplus_installed()
            
            # Generate solutions
            solutions_file = self._generate_solutions_humaneval()
            
            # Run evaluation
            results = self._run_evalplus_evaluation(solutions_file, "humaneval")
            
            return results
            
        except Exception as e:
            logger.error(f"HumanEval-Plus evaluation failed: {e}")
            # Return fallback results
            return {'pass@1': 0.0, 'pass@10': 0.0}
    
    def evaluate_mbpp_plus(self) -> Dict[str, float]:
        """Evaluate on MBPP-Plus benchmark."""
        try:
            # Install evalplus if not available
            self._ensure_evalplus_installed()
            
            # Generate solutions
            solutions_file = self._generate_solutions_mbpp()
            
            # Run evaluation
            results = self._run_evalplus_evaluation(solutions_file, "mbpp")
            
            return results
            
        except Exception as e:
            logger.error(f"MBPP-Plus evaluation failed: {e}")
            # Return fallback results
            return {'pass@1': 0.0, 'pass@10': 0.0}
    
    def _ensure_evalplus_installed(self):
        """Ensure evalplus is installed."""
        try:
            import evalplus
        except ImportError:
            logger.info("Installing evalplus...")
            subprocess.check_call(["pip", "install", "evalplus"])
    
    def _generate_solutions_humaneval(self) -> str:
        """Generate solutions for HumanEval problems."""
        try:
            from evalplus.data import get_human_eval_plus
            
            problems = get_human_eval_plus()
            solutions = []
            
            for task_id, problem in problems.items():
                logger.info(f"Generating solution for {task_id}")
                
                # Create prompt
                prompt = self._create_code_prompt(problem['prompt'])
                
                # Generate multiple solutions
                task_solutions = []
                for i in range(self.num_samples):
                    solution = self._generate_code(prompt)
                    task_solutions.append(solution)
                
                solutions.append({
                    'task_id': task_id,
                    'completion': task_solutions[0],  # Use first solution for pass@1
                    'all_completions': task_solutions,
                })
            
            # Save solutions to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for solution in solutions:
                    json.dump(solution, f)
                    f.write('\n')
                return f.name
                
        except Exception as e:
            logger.error(f"Failed to generate HumanEval solutions: {e}")
            raise
    
    def _generate_solutions_mbpp(self) -> str:
        """Generate solutions for MBPP problems."""
        try:
            from evalplus.data import get_mbpp_plus
            
            problems = get_mbpp_plus()
            solutions = []
            
            for task_id, problem in problems.items():
                logger.info(f"Generating solution for {task_id}")
                
                # Create prompt
                prompt = self._create_code_prompt(problem['prompt'])
                
                # Generate multiple solutions
                task_solutions = []
                for i in range(self.num_samples):
                    solution = self._generate_code(prompt)
                    task_solutions.append(solution)
                
                solutions.append({
                    'task_id': task_id,
                    'completion': task_solutions[0],  # Use first solution for pass@1
                    'all_completions': task_solutions,
                })
            
            # Save solutions to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                for solution in solutions:
                    json.dump(solution, f)
                    f.write('\n')
                return f.name
                
        except Exception as e:
            logger.error(f"Failed to generate MBPP solutions: {e}")
            raise
    
    def _create_code_prompt(self, problem_prompt: str) -> str:
        """Create code generation prompt."""
        return f"""Please complete the following Python function:

{problem_prompt}

Complete the function implementation:
```python"""
    
    def _generate_code(self, prompt: str) -> str:
        """Generate code using the model."""
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    tokenizer=self.tokenizer,
                    stop_strings=["```", "\n\n\n"],
                )
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return ""
    
    def _run_evalplus_evaluation(self, solutions_file: str, dataset: str) -> Dict[str, float]:
        """Run evalplus evaluation on solutions file."""
        try:
            # Run evalplus command
            cmd = [
                "evalplus.evaluate",
                "--dataset", dataset,
                "--samples", solutions_file,
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Evalplus evaluation failed: {result.stderr}")
                return {'pass@1': 0.0, 'pass@10': 0.0}
            
            # Parse results
            output_lines = result.stdout.strip().split('\n')
            results = {}
            
            for line in output_lines:
                if 'pass@1' in line:
                    try:
                        results['pass@1'] = float(line.split()[-1])
                    except:
                        results['pass@1'] = 0.0
                elif 'pass@10' in line:
                    try:
                        results['pass@10'] = float(line.split()[-1])
                    except:
                        results['pass@10'] = 0.0
            
            # Clean up temporary file
            os.unlink(solutions_file)
            
            return results
            
        except Exception as e:
            logger.error(f"Evalplus evaluation failed: {e}")
            return {'pass@1': 0.0, 'pass@10': 0.0}


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run EvalPlus benchmarks")
    parser.add_argument("checkpoint_path", help="Path to model checkpoint")
    parser.add_argument("--output", "-o", default="eval_results.json", 
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Run evaluation
    runner = EvalPlusRunner(
        model_path="Qwen/Qwen2.5-3B",
        lora_path=args.checkpoint_path,
    )
    
    results = runner.run_evaluation()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {args.output}")


if __name__ == "__main__":
    main()