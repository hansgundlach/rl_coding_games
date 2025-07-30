#!/usr/bin/env python3
"""
MBPP (Mostly Basic Python Problems) Evaluator
Compatible with MIT Supercloud V100 (offline) and Lambda A100 environments
"""

import json
import os
import sys
import tempfile
import subprocess
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import random

# Import vLLM integration
try:
    from utils.vllm_client import get_vllm_integration
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    get_vllm_integration = lambda: None

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for MBPP evaluation."""
    
    # Evaluation control
    enabled: bool = True
    num_questions: int = 10
    eval_at_start: bool = True
    eval_at_end: bool = True
    eval_interval_steps: Optional[int] = None  # If None, only eval at start/end
    
    # Dataset configuration
    dataset_path: Optional[str] = None  # Path to local MBPP dataset
    use_sanitized: bool = True  # Use sanitized version (427 problems) vs full (974)
    test_split: str = "test"  # Which split to use: "test", "validation", "train"
    
    # Execution settings
    timeout_seconds: int = 10
    max_memory_mb: int = 128
    safe_execution: bool = True
    
    # Model settings
    max_new_tokens: int = 512
    temperature: float = 0.2
    do_sample: bool = True
    
    # Output settings
    save_results: bool = True
    results_dir: str = "./eval_results"
    verbose: bool = True


class MBPPEvaluator:
    """MBPP evaluator with offline support and configurable evaluation."""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.problems = []
        self.step_count = 0
        self.eval_history = []
        
        # Setup results directory
        if self.config.save_results:
            os.makedirs(self.config.results_dir, exist_ok=True)
        
        # Load dataset if enabled
        if self.config.enabled:
            self._load_dataset()
    
    def _load_dataset(self):
        """Load MBPP dataset from local files."""
        if not self.config.dataset_path:
            # Try to find dataset in common locations
            possible_paths = [
                "./evaluation/datasets/sanitized-mbpp.json",
                "./evaluation/datasets/mbpp.jsonl",
                "./data/mbpp/sanitized-mbpp.json",
                "./data/mbpp/mbpp.jsonl", 
                "../data/mbpp/sanitized-mbpp.json",
                "../data/mbpp/mbpp.jsonl",
                "/home/gridsan/hgundlach/game_project_rl/evaluation/datasets/sanitized-mbpp.json"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.config.dataset_path = path
                    break
        
        if not self.config.dataset_path or not os.path.exists(self.config.dataset_path):
            logger.warning(f"MBPP dataset not found at {self.config.dataset_path}")
            logger.warning("Evaluation will be disabled. Download MBPP dataset first.")
            self.config.enabled = False
            return
        
        try:
            # Load dataset based on file format
            if self.config.dataset_path.endswith('.jsonl'):
                # Original MBPP format
                with open(self.config.dataset_path, 'r') as f:
                    self.problems = [json.loads(line) for line in f if line.strip()]
            else:
                # Sanitized JSON format
                with open(self.config.dataset_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.problems = data
                    else:
                        self.problems = data.get('problems', [])
            
            # Filter to test split if specified
            if self.config.test_split == "test":
                # Use problems 11-510 for testing (standard split)
                self.problems = [p for p in self.problems if 11 <= p.get('task_id', 0) <= 510]
            elif self.config.test_split == "validation":
                # Use problems 511-600 for validation
                self.problems = [p for p in self.problems if 511 <= p.get('task_id', 0) <= 600]
            
            logger.info(f"Loaded {len(self.problems)} MBPP problems from {self.config.dataset_path}")
            
        except Exception as e:
            logger.error(f"Failed to load MBPP dataset: {e}")
            self.config.enabled = False
    
    def should_evaluate(self, is_start: bool = False, is_end: bool = False) -> bool:
        """Check if evaluation should run at this step."""
        if not self.config.enabled:
            return False
            
        if is_start and self.config.eval_at_start:
            return True
            
        if is_end and self.config.eval_at_end:
            return True
            
        if (self.config.eval_interval_steps and 
            self.step_count > 0 and 
            self.step_count % self.config.eval_interval_steps == 0):
            return True
            
        return False
    
    def increment_step(self):
        """Increment step counter for interval-based evaluation."""
        self.step_count += 1
    
    def get_eval_problems(self) -> List[Dict]:
        """Get subset of problems for evaluation."""
        if not self.problems:
            return []
        
        # Randomly sample problems for this evaluation
        num_to_sample = min(self.config.num_questions, len(self.problems))
        return random.sample(self.problems, num_to_sample)
    
    def create_prompt(self, problem: Dict) -> str:
        """Create evaluation prompt from MBPP problem."""
        # Handle both original and sanitized formats
        if 'text' in problem:
            description = problem['text']
        elif 'prompt' in problem:
            description = problem['prompt']
        else:
            description = "Write a Python function to solve the problem."
        
        # Get first test case as example
        test_list = problem.get('test_list', [])
        if test_list:
            example = test_list[0]
            prompt = f'"""\n{description}\n{example}\n"""\n'
        else:
            prompt = f'"""\n{description}\n"""\n'
        
        return prompt
    
    def extract_code(self, completion: str) -> str:
        """Extract Python code from model completion."""
        # Try to find code in markdown blocks first
        import re
        
        code_match = re.search(r'```python\s*\n(.*?)```', completion, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # If no markdown, use the completion as-is
        return completion.strip()
    
    def safe_execute_code(self, code: str, test_cases: List[str]) -> Dict[str, Any]:
        """Safely execute code with test cases."""
        result = {
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0.0,
            'timeout': False
        }
        
        if not code.strip():
            result['error'] = 'Empty code'
            return result
        
        # Combine code with test cases
        full_code = code + '\n\n' + '\n'.join(test_cases)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            start_time = time.time()
            
            # Execute in subprocess with timeout and memory limits
            env = os.environ.copy()
            env['PYTHONPATH'] = ''  # Minimal environment
            
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tempfile.gettempdir(),
                env=env
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.config.timeout_seconds)
                result['execution_time'] = time.time() - start_time
                result['output'] = stdout.strip()
                result['error'] = stderr.strip()
                result['success'] = process.returncode == 0 and not stderr
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                result['timeout'] = True
                result['error'] = f'Execution timed out after {self.config.timeout_seconds}s'
                result['execution_time'] = self.config.timeout_seconds
                
        except Exception as e:
            result['error'] = f'Execution error: {str(e)}'
            result['execution_time'] = time.time() - start_time
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return result
    
    def evaluate_single_problem(self, problem: Dict, model, tokenizer, use_vllm: bool = False) -> Dict[str, Any]:
        """Evaluate model on a single MBPP problem with optional vLLM acceleration."""
        prompt = self.create_prompt(problem)
        
        try:
            # Choose generation method
            if use_vllm and VLLM_AVAILABLE:
                generated_text = self._generate_with_vllm(prompt)
            else:
                generated_text = self._generate_with_hf(prompt, model, tokenizer)
            
            # Extract code
            code = self.extract_code(generated_text)
            
            # Get test cases
            test_cases = problem.get('test_list', [])
            
            # Execute and evaluate
            exec_result = self.safe_execute_code(code, test_cases)
            
            return {
                'task_id': problem.get('task_id', 0),
                'prompt': prompt,
                'generated_code': code,
                'full_completion': generated_text,
                'execution_result': exec_result,
                'passed': exec_result['success']
            }
            
        except Exception as e:
            return {
                'task_id': problem.get('task_id', 0),
                'prompt': prompt,
                'generated_code': '',
                'full_completion': '',
                'execution_result': {'success': False, 'error': str(e)},
                'passed': False
            }
    
    def _generate_with_vllm(self, prompt: str) -> str:
        """Generate text using vLLM server"""
        vllm_integration = get_vllm_integration()
        if not vllm_integration:
            raise RuntimeError("vLLM integration not available")
            
        completions = vllm_integration.generate_completions([prompt], mode="mbpp_evaluation")
        return completions[0] if completions else ""
    
    def _generate_with_hf(self, prompt: str, model, tokenizer) -> str:
        """Generate text using HuggingFace model"""
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return generated_text
    
    def evaluate_model(self, model, tokenizer, step: int = 0, phase: str = "eval") -> Dict[str, Any]:
        """Run full MBPP evaluation on model with optional vLLM acceleration."""
        if not self.config.enabled:
            return {"enabled": False}
        
        start_time = time.time()
        problems = self.get_eval_problems()
        
        # Check if vLLM should be used for evaluation
        vllm_integration = get_vllm_integration()
        use_vllm = (vllm_integration and 
                   vllm_integration.vllm_config.enabled and 
                   vllm_integration.vllm_config.integration.get('use_for_evaluation', False))
        
        if use_vllm:
            logger.info(f"🚀 Running MBPP evaluation with vLLM at step {step} ({phase}) on {len(problems)} problems")
        else:
            logger.info(f"Running MBPP evaluation with HuggingFace at step {step} ({phase}) on {len(problems)} problems")
        
        results = []
        passed_count = 0
        
        for i, problem in enumerate(problems):
            if self.config.verbose:
                logger.info(f"Evaluating problem {i+1}/{len(problems)}: task_id={problem.get('task_id', 'unknown')}")
            
            result = self.evaluate_single_problem(problem, model, tokenizer, use_vllm=use_vllm)
            results.append(result)
            
            if result['passed']:
                passed_count += 1
            
            # Log progress
            if self.config.verbose and i < 3:  # Show first few
                logger.info(f"Problem {i+1} result: {'PASSED' if result['passed'] else 'FAILED'}")
                if not result['passed']:
                    logger.info(f"Error: {result['execution_result'].get('error', 'unknown')}")
        
        # Calculate metrics
        total_problems = len(problems)
        pass_rate = passed_count / total_problems if total_problems > 0 else 0.0
        eval_time = time.time() - start_time
        
        eval_summary = {
            'step': step,
            'phase': phase,
            'timestamp': time.time(),
            'total_problems': total_problems,
            'problems_passed': passed_count,
            'pass_rate': pass_rate,
            'eval_time_seconds': eval_time,
            'config': {
                'num_questions': self.config.num_questions,
                'temperature': self.config.temperature,
                'max_new_tokens': self.config.max_new_tokens,
                'dataset_path': self.config.dataset_path
            }
        }
        
        # Save detailed results if enabled
        if self.config.save_results:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            
            # Save summary
            summary_file = os.path.join(
                self.config.results_dir, 
                f"mbpp_summary_step_{step}_{phase}_{timestamp_str}.json"
            )
            with open(summary_file, 'w') as f:
                json.dump(eval_summary, f, indent=2)
            
            # Save detailed results
            details_file = os.path.join(
                self.config.results_dir,
                f"mbpp_details_step_{step}_{phase}_{timestamp_str}.json"
            )
            with open(details_file, 'w') as f:
                json.dump({
                    'summary': eval_summary,
                    'detailed_results': results
                }, f, indent=2)
        
        # Add to history
        self.eval_history.append(eval_summary)
        
        logger.info(f"MBPP evaluation completed: {passed_count}/{total_problems} passed ({pass_rate:.3f}) in {eval_time:.1f}s")
        
        return eval_summary


def create_default_config(**kwargs) -> EvalConfig:
    """Create default evaluation config with optional overrides."""
    return EvalConfig(**kwargs)


def download_mbpp_dataset(save_dir: str = "./data/mbpp") -> bool:
    """Download MBPP dataset for offline use."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Try using datasets library first
        try:
            from datasets import load_dataset
            
            logger.info("Downloading MBPP dataset using datasets library...")
            
            # Download sanitized version (recommended)
            dataset = load_dataset("google-research-datasets/mbpp", "sanitized", cache_dir=save_dir)
            
            # Save as JSON for easy loading
            sanitized_path = os.path.join(save_dir, "sanitized-mbpp.json")
            with open(sanitized_path, 'w') as f:
                json.dump(dataset['test'].to_list(), f, indent=2)
            
            logger.info(f"MBPP dataset saved to {sanitized_path}")
            return True
            
        except ImportError:
            logger.warning("datasets library not available, trying direct download...")
            
        # Fallback to direct download
        import urllib.request
        
        urls = [
            ("https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json", "sanitized-mbpp.json"),
            ("https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl", "mbpp.jsonl")
        ]
        
        for url, filename in urls:
            try:
                filepath = os.path.join(save_dir, filename)
                logger.info(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                logger.info(f"Saved {filename} to {filepath}")
            except Exception as e:
                logger.warning(f"Failed to download {filename}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download MBPP dataset: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    config = EvalConfig(
        enabled=True,
        num_questions=5,
        dataset_path="./data/mbpp/sanitized-mbpp.json"
    )
    
    evaluator = MBPPEvaluator(config)
    
    if not evaluator.config.enabled:
        print("Dataset not found. Attempting to download...")
        if download_mbpp_dataset():
            print("Dataset downloaded successfully. Please run again.")
        else:
            print("Failed to download dataset. Please download manually.")