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
from concurrent.futures import ThreadPoolExecutor

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
    consistent_questions: bool = True  # Use same questions for all interval evaluations

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
    
    # Parallel execution settings
    parallel_execution: bool = True
    num_workers: Optional[int] = None  # Defaults to number of CPU cores
    batch_generation: bool = True  # Enable batch generation for faster evaluation
    
    # Batch size settings
    max_batch_size: Optional[int] = None  # Maximum batch size (auto-detect if None)
    adaptive_batching: bool = True  # Automatically adjust batch size based on memory


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
                "/home/gridsan/hgundlach/game_project_rl/evaluation/datasets/sanitized-mbpp.json",
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
            if self.config.dataset_path.endswith(".jsonl"):
                # Original MBPP format
                with open(self.config.dataset_path, "r") as f:
                    self.problems = [json.loads(line) for line in f if line.strip()]
            else:
                # Sanitized JSON format
                with open(self.config.dataset_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.problems = data
                    else:
                        self.problems = data.get("problems", [])

            # Filter to test split if specified
            if self.config.test_split == "test":
                # Use problems 11-510 for testing (standard split)
                self.problems = [
                    p for p in self.problems if 11 <= p.get("task_id", 0) <= 510
                ]
            elif self.config.test_split == "validation":
                # Use problems 511-600 for validation
                self.problems = [
                    p for p in self.problems if 511 <= p.get("task_id", 0) <= 600
                ]

            logger.info(
                f"Loaded {len(self.problems)} MBPP problems from {self.config.dataset_path}"
            )

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

        if (
            self.config.eval_interval_steps
            and self.step_count > 0
            and self.step_count % self.config.eval_interval_steps == 0
        ):
            return True

        return False

    def increment_step(self):
        """Increment step counter for interval-based evaluation."""
        self.step_count += 1

    def get_eval_problems(self) -> List[Dict]:
        """Get subset of problems for evaluation."""
        if not self.problems:
            return []

        num_to_sample = min(self.config.num_questions, len(self.problems))

        # Check if we should use consistent questions across evaluations
        if (
            hasattr(self.config, "consistent_questions")
            and self.config.consistent_questions
        ):
            # Use deterministic selection - sort by task_id and take first N
            sorted_problems = sorted(self.problems, key=lambda p: p.get("task_id", 0))
            return sorted_problems[:num_to_sample]
        else:
            # Use random sampling (original behavior)
            return random.sample(self.problems, num_to_sample)

    def create_prompt(self, problem: Dict) -> str:
        """Create evaluation prompt from MBPP problem."""
        # Handle both original and sanitized formats
        if "text" in problem:
            description = problem["text"]
        elif "prompt" in problem:
            description = problem["prompt"]
        else:
            description = "Write a Python function to solve the problem."

        # Get first test case as example
        test_list = problem.get("test_list", [])
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

        code_match = re.search(r"```python\s*\n(.*?)```", completion, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # If no markdown, use the completion as-is
        return completion.strip()

    def safe_execute_code(self, code: str, test_cases: List[str]) -> Dict[str, Any]:
        """Safely execute code with test cases."""
        result = {
            "success": False,
            "output": "",
            "error": "",
            "execution_time": 0.0,
            "timeout": False,
        }

        if not code.strip():
            result["error"] = "Empty code"
            return result

        # Combine code with test cases
        full_code = code + "\n\n" + "\n".join(test_cases)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            temp_file = f.name

        try:
            start_time = time.time()

            # Execute in subprocess with timeout and memory limits
            env = os.environ.copy()
            env["PYTHONPATH"] = ""  # Minimal environment

            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tempfile.gettempdir(),
                env=env,
            )

            try:
                stdout, stderr = process.communicate(
                    timeout=self.config.timeout_seconds
                )
                result["execution_time"] = time.time() - start_time
                result["output"] = stdout.strip()
                result["error"] = stderr.strip()
                result["success"] = process.returncode == 0 and not stderr

            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                result["timeout"] = True
                result["error"] = (
                    f"Execution timed out after {self.config.timeout_seconds}s"
                )
                result["execution_time"] = self.config.timeout_seconds

        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
            result["execution_time"] = time.time() - start_time

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

        return result

    def evaluate_single_problem(
        self, problem: Dict, model, tokenizer, use_vllm: bool = False
    ) -> Dict[str, Any]:
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
            test_cases = problem.get("test_list", [])

            # Execute and evaluate
            exec_result = self.safe_execute_code(code, test_cases)

            return {
                "task_id": problem.get("task_id", 0),
                "prompt": prompt,
                "generated_code": code,
                "full_completion": generated_text,
                "execution_result": exec_result,
                "passed": exec_result["success"],
            }

        except Exception as e:
            return {
                "task_id": problem.get("task_id", 0),
                "prompt": prompt,
                "generated_code": "",
                "full_completion": "",
                "execution_result": {"success": False, "error": str(e)},
                "passed": False,
            }

    def _generate_with_vllm(self, prompt: str) -> str:
        """Generate text using vLLM server"""
        vllm_integration = get_vllm_integration()
        if not vllm_integration:
            raise RuntimeError("vLLM integration not available")

        completions = vllm_integration.generate_completions(
            [prompt], mode="mbpp_evaluation"
        )
        return completions[0] if completions else ""

    def _generate_with_hf(self, prompt: str, model, tokenizer) -> str:
        """Generate text using HuggingFace model"""
        logger.info(
            f"🔍 HF Generate config: temperature={self.config.temperature}, do_sample={self.config.do_sample}"
        )

        # Tokenize
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024, padding=True
        )

        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Prepare generation parameters with proper temperature/do_sample handling
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        # Handle temperature=0.0 case properly for newer transformers
        if self.config.temperature == 0.0:
            logger.info(f"🎯 Using greedy decoding (temperature=0.0, do_sample=False)")
            generation_kwargs["do_sample"] = False  # Greedy decoding
            # Don't add temperature for greedy decoding
        else:
            logger.info(
                f"🎯 Using sampling (temperature={self.config.temperature}, do_sample={self.config.do_sample})"
            )
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["do_sample"] = self.config.do_sample

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        return generated_text

    def _batch_generate_with_hf(
        self, prompts: List[str], model, tokenizer
    ) -> List[str]:
        """Batch generate text using HuggingFace model - MUCH faster than individual calls."""
        if not prompts:
            return []

        if len(prompts) == 1:
            return [self._generate_with_hf(prompts[0], model, tokenizer)]

        # Batch tokenize all prompts
        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True, max_length=1024, padding=True
        )

        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Prepare generation parameters with proper temperature/do_sample handling
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        # Handle temperature=0.0 case properly for newer transformers
        if self.config.temperature == 0.0:
            generation_kwargs["do_sample"] = False  # Greedy decoding
            # Don't add temperature for greedy decoding
        else:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["do_sample"] = self.config.do_sample

        # Batch generate for all prompts at once (HUGE SPEEDUP!)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        # Decode all completions
        generated_texts = []
        for i, output in enumerate(outputs):
            generated_text = tokenizer.decode(
                output[inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            generated_texts.append(generated_text)

        logger.info(
            f"🚀 Batch generated {len(prompts)} MBPP completions in one GPU call!"
        )
        return generated_texts

    def _batch_generate_with_vllm(self, prompts: List[str]) -> List[str]:
        """Batch generate text using vLLM server - even faster with proper batching."""
        vllm_integration = get_vllm_integration()
        if not vllm_integration:
            raise RuntimeError("vLLM integration not available")

        # vLLM already handles batching efficiently
        completions = vllm_integration.generate_completions(
            prompts, mode="mbpp_evaluation"
        )

        logger.info(f"🚀 vLLM batch generated {len(prompts)} MBPP completions!")
        return completions if completions else [""] * len(prompts)

    def _get_adaptive_batch_size(self, total_problems: int) -> int:
        """Get adaptive batch size based on GPU memory and problem count."""
        if not torch.cuda.is_available():
            return 1
        
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        # If user specified max_batch_size, respect it
        if self.config.max_batch_size is not None:
            return min(self.config.max_batch_size, total_problems)
        
        # Adaptive batch sizing based on GPU memory
        if gpu_memory >= 30:  # V100 or larger
            # Conservative batch sizes for V100
            if total_problems <= 10:
                return total_problems  # Small batches, process all at once
            elif total_problems <= 50:
                return 10  # Medium batches
            else:
                return 5  # Large problem sets, small batches
        elif gpu_memory >= 20:  # A100 or similar
            # More aggressive for A100
            if total_problems <= 20:
                return total_problems
            elif total_problems <= 100:
                return 20
            else:
                return 10
        else:
            # Smaller GPUs
            return min(5, total_problems)
    
    def _batch_generate_with_hf_adaptive(self, prompts: List[str], model, tokenizer) -> List[str]:
        """Batch generate text using HuggingFace model with adaptive batching."""
        if not prompts:
            return []
            
        if len(prompts) == 1:
            return [self._generate_with_hf(prompts[0], model, tokenizer)]
        
        # Determine optimal batch size
        optimal_batch_size = self._get_adaptive_batch_size(len(prompts))
        logger.info(f"🎯 Adaptive batching: processing {len(prompts)} problems in batches of {optimal_batch_size}")
        
        all_generated_texts = []
        total_batches = (len(prompts) + optimal_batch_size - 1) // optimal_batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * optimal_batch_size
            end_idx = min(start_idx + optimal_batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]
            
            logger.info(f"🎯 Processing batch {batch_idx + 1}/{total_batches} ({len(batch_prompts)} problems)")
            
            try:
                # Process this batch
                batch_texts = self._batch_generate_with_hf(batch_prompts, model, tokenizer)
                all_generated_texts.extend(batch_texts)
                
                logger.info(f"✅ Batch {batch_idx + 1} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Batch {batch_idx + 1} failed: {e}")
                logger.info("🔄 Falling back to individual generation for this batch...")
                
                # Fall back to individual generation for this batch
                for prompt in batch_prompts:
                    try:
                        text = self._generate_with_hf(prompt, model, tokenizer)
                        all_generated_texts.append(text)
                    except Exception as e2:
                        logger.error(f"❌ Individual generation failed: {e2}")
                        all_generated_texts.append("")
        
        logger.info(f"🏁 Adaptive batching complete: {len(all_generated_texts)} total completions")
        return all_generated_texts
    
    def _batch_generate_with_vllm_adaptive(self, prompts: List[str]) -> List[str]:
        """Batch generate text using vLLM server with adaptive batching."""
        vllm_integration = get_vllm_integration()
        if not vllm_integration:
            raise RuntimeError("vLLM integration not available")

        if not prompts:
            return []
            
        if len(prompts) == 1:
            return [self._generate_with_vllm(prompts[0])]
        
        # Determine optimal batch size for vLLM
        optimal_batch_size = self._get_adaptive_batch_size(len(prompts))
        logger.info(f"🎯 vLLM adaptive batching: processing {len(prompts)} problems in batches of {optimal_batch_size}")
        
        all_generated_texts = []
        total_batches = (len(prompts) + optimal_batch_size - 1) // optimal_batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * optimal_batch_size
            end_idx = min(start_idx + optimal_batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]
            
            logger.info(f"🚀 vLLM processing batch {batch_idx + 1}/{total_batches} ({len(batch_prompts)} problems)")
            
            try:
                # vLLM handles batching efficiently
                batch_completions = vllm_integration.generate_completions(
                    batch_prompts, mode="mbpp_evaluation"
                )
                all_generated_texts.extend(batch_completions if batch_completions else [""] * len(batch_prompts))
                
                logger.info(f"✅ vLLM batch {batch_idx + 1} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ vLLM batch {batch_idx + 1} failed: {e}")
                logger.info("🔄 Falling back to individual generation for this batch...")
                
                # Fall back to individual generation for this batch
                for prompt in batch_prompts:
                    try:
                        text = self._generate_with_vllm(prompt)
                        all_generated_texts.append(text)
                    except Exception as e2:
                        logger.error(f"❌ Individual vLLM generation failed: {e2}")
                        all_generated_texts.append("")
        
        logger.info(f"🏁 vLLM adaptive batching complete: {len(all_generated_texts)} total completions")
        return all_generated_texts

    def evaluate_model(
        self, model, tokenizer, step: int = 0, phase: str = "eval"
    ) -> Dict[str, Any]:
        """Run full MBPP evaluation on model with optional vLLM acceleration.
        This implementation parallelises the CPU-bound *safe execution* phase using a
        ThreadPool so that multiple test cases can be verified simultaneously.
        Generation (GPU / vLLM) is still done sequentially to avoid GPU contention.
        """
        if not self.config.enabled:
            return {"enabled": False}

        start_time = time.time()
        problems = self.get_eval_problems()

        # Determine whether to use vLLM for generation
        vllm_integration = get_vllm_integration()
        use_vllm = (
            vllm_integration
            and vllm_integration.vllm_config.enabled
            and vllm_integration.vllm_config.integration.get(
                "use_for_evaluation", False
            )
        )

        if use_vllm:
            logger.info(
                f"🚀 Running MBPP evaluation with vLLM at step {step} ({phase}) on {len(problems)} problems"
            )
        else:
            logger.info(
                f"Running MBPP evaluation with HuggingFace at step {step} ({phase}) on {len(problems)} problems"
            )

        # ------------------------------------------------------------------
        # 1. Prompt construction & batch generation (MUCH faster!)
        # ------------------------------------------------------------------
        generation_start_time = time.time()
        logger.info(f"🚀 Starting generation for {len(problems)} problems...")

        # Build all prompts at once
        prompt_start = time.time()
        prompts = []
        for problem in problems:
            prompt = self.create_prompt(problem)
            prompts.append(prompt)
        prompt_time = time.time() - prompt_start
        logger.info(f"📝 Built {len(prompts)} prompts in {prompt_time:.3f}s")

        # Batch generate all completions (if enabled)
        generated_texts = []
        use_batch = self.config.batch_generation and len(problems) > 1
        
        if use_batch:
            batch_start = time.time()
            try:
                if use_vllm:
                    if self.config.adaptive_batching:
                        generated_texts = self._batch_generate_with_vllm_adaptive(prompts)
                    else:
                        generated_texts = self._batch_generate_with_vllm(prompts)
                else:
                    if self.config.adaptive_batching:
                        generated_texts = self._batch_generate_with_hf_adaptive(prompts, model, tokenizer)
                    else:
                        generated_texts = self._batch_generate_with_hf(prompts, model, tokenizer)
                
                batch_time = time.time() - batch_start
                avg_per_problem = batch_time / len(problems)
                logger.info(f"🚀 BATCH GENERATION SUCCESS: {len(generated_texts)} completions in {batch_time:.3f}s ({avg_per_problem:.3f}s per problem)")
                use_batch = True  # Success, don't fall back
                    
            except Exception as e:
                batch_time = time.time() - batch_start
                logger.error(f"❌ BATCH GENERATION FAILED after {batch_time:.3f}s: {e}")
                logger.info("🔄 Falling back to individual generation...")
                use_batch = False  # Fall back to individual

        # Individual generation (fallback or if batch disabled)
        if not use_batch:
            individual_start = time.time()
            if not self.config.batch_generation:
                logger.info("🔄 Using individual generation (batch disabled in config)")

            for idx, problem in enumerate(problems):
                problem_start = time.time()
                if self.config.verbose:
                    logger.info(
                        f"🔄 Generating completion for problem {idx + 1}/{len(problems)} (task_id={problem.get('task_id', 'unknown')})"
                    )

                prompt = prompts[idx]
                try:
                    generated_text = (
                        self._generate_with_vllm(prompt)
                        if use_vllm
                        else self._generate_with_hf(prompt, model, tokenizer)
                    )
                    problem_time = time.time() - problem_start
                    if self.config.verbose:
                        logger.info(
                            f"✅ Problem {idx + 1} completed in {problem_time:.3f}s"
                        )
                except Exception as e:
                    problem_time = time.time() - problem_start
                    error_msg = str(e)
                    task_id = problem.get("task_id", "unknown")
                    logger.error(
                        f"❌ Problem {idx + 1} failed after {problem_time:.3f}s - task_id={task_id}: {error_msg}"
                    )

                    # Provide specific guidance for common errors
                    if "temperature" in error_msg and "strictly positive" in error_msg:
                        logger.error(
                            "💡 SOLUTION: Set 'do_sample: false' in config when using temperature=0.0 for greedy decoding"
                        )
                    elif "do_sample" in error_msg:
                        logger.error(
                            "💡 SOLUTION: Check your generation config - ensure do_sample and temperature are compatible"
                        )

                    generated_text = ""

                generated_texts.append(generated_text)

            individual_time = time.time() - individual_start
            avg_per_problem = individual_time / len(problems)
            logger.info(
                f"🔄 INDIVIDUAL GENERATION COMPLETE: {len(problems)} problems in {individual_time:.3f}s ({avg_per_problem:.3f}s per problem)"
            )

        total_generation_time = time.time() - generation_start_time
        logger.info(f"⚡ GENERATION PHASE COMPLETE: {total_generation_time:.3f}s total")

        # Build generation records
        generation_records = []
        for problem, prompt, generated_text in zip(problems, prompts, generated_texts):
            code = self.extract_code(generated_text)
            test_cases = problem.get("test_list", [])
            generation_records.append(
                (problem, prompt, code, generated_text, test_cases)
            )

        # ------------------------------------------------------------------
        # 2. Safe execution of generated code  (CPU-bound, parallel)
        # ------------------------------------------------------------------
        execution_start_time = time.time()
        logger.info(
            f"🔧 Starting code execution phase for {len(generation_records)} problems..."
        )

        if self.config.parallel_execution and len(generation_records) > 1:
            max_workers = self.config.num_workers or (os.cpu_count() or 1)
        else:
            max_workers = 1  # Fallback to sequential execution

        def _exec_wrapper(record):
            _problem, _prompt, _code, _gen_text, _tests = record
            return self.safe_execute_code(_code, _tests)

        if max_workers > 1:
            logger.info(
                f"⚙️  Executing test cases in parallel with {max_workers} workers"
            )
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                exec_results = list(pool.map(_exec_wrapper, generation_records))
        else:
            logger.info("🔧 Executing test cases sequentially")
            exec_results = [_exec_wrapper(r) for r in generation_records]

        execution_time = time.time() - execution_start_time
        avg_exec_time = execution_time / len(generation_records)
        logger.info(
            f"⚡ EXECUTION PHASE COMPLETE: {execution_time:.3f}s total ({avg_exec_time:.3f}s per problem)"
        )

        # ------------------------------------------------------------------
        # 3. Assemble results & metrics
        # ------------------------------------------------------------------
        results = []
        passed_count = 0
        generation_failures = 0
        execution_failures = 0

        for record, exec_result in zip(generation_records, exec_results):
            problem, prompt, code, full_completion, _tests = record
            passed = exec_result.get("success", False)

            # Track failure types for better diagnostics
            if not full_completion:  # Generation failed
                generation_failures += 1
            elif not passed:  # Generation succeeded but execution failed
                execution_failures += 1

            results.append(
                {
                    "task_id": problem.get("task_id", 0),
                    "prompt": prompt,
                    "generated_code": code,
                    "full_completion": full_completion,
                    "execution_result": exec_result,
                    "passed": passed,
                }
            )
            if passed:
                passed_count += 1

        total_problems = len(problems)
        pass_rate = passed_count / total_problems if total_problems > 0 else 0.0
        eval_time = time.time() - start_time

        eval_summary = {
            "step": step,
            "phase": phase,
            "timestamp": time.time(),
            "total_problems": total_problems,
            "problems_passed": passed_count,
            "pass_rate": pass_rate,
            "eval_time_seconds": eval_time,
            "config": {
                "num_questions": self.config.num_questions,
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_new_tokens,
                "dataset_path": self.config.dataset_path,
                "parallel_execution": self.config.parallel_execution,
                "num_workers": max_workers,
            },
        }

        # Save results if enabled
        if self.config.save_results:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            summary_file = os.path.join(
                self.config.results_dir,
                f"mbpp_summary_step_{step}_{phase}_{timestamp_str}.json",
            )
            details_file = os.path.join(
                self.config.results_dir,
                f"mbpp_details_step_{step}_{phase}_{timestamp_str}.json",
            )
            try:
                with open(summary_file, "w") as f:
                    json.dump(eval_summary, f, indent=2)
                with open(details_file, "w") as f:
                    json.dump(
                        {"summary": eval_summary, "detailed_results": results},
                        f,
                        indent=2,
                    )
            except Exception as e:
                logger.error(f"Failed to save evaluation results: {e}")

        self.eval_history.append(eval_summary)

        # ------------------------------------------------------------------
        # 4. Final evaluation summary with comprehensive timing
        # ------------------------------------------------------------------
        summary_start_time = time.time()

        # Add timing breakdown to eval_summary
        eval_summary.update(
            {
                "timing_breakdown": {
                    "total_eval_time": eval_time,
                    "generation_time": total_generation_time,
                    "execution_time": execution_time,
                    "batch_generation_used": use_batch,
                    "parallel_execution_used": max_workers > 1,
                    "num_workers": max_workers,
                    "avg_generation_per_problem": (
                        total_generation_time / len(problems) if problems else 0
                    ),
                    "avg_execution_per_problem": (
                        execution_time / len(problems) if problems else 0
                    ),
                }
            }
        )

        # Enhanced logging with failure breakdown and detailed timing
        logger.info(
            f"🏁 MBPP EVALUATION COMPLETE: {passed_count}/{total_problems} passed ({pass_rate:.3f}) in {eval_time:.1f}s"
        )

        # Detailed timing breakdown log
        logger.info("📊 TIMING BREAKDOWN:")
        logger.info(
            f"   Generation: {total_generation_time:.3f}s ({100*total_generation_time/eval_time:.1f}%)"
        )
        logger.info(
            f"   Execution:  {execution_time:.3f}s ({100*execution_time/eval_time:.1f}%)"
        )
        other_time = eval_time - total_generation_time - execution_time
        logger.info(
            f"   Other:      {other_time:.3f}s ({100*other_time/eval_time:.1f}%)"
        )

        # Performance summary
        if use_batch:
            logger.info(
                f"⚡ BATCH GENERATION: Processed {len(problems)} problems in single GPU call"
            )
        else:
            logger.info(
                f"🔄 INDIVIDUAL GENERATION: {len(problems)} sequential GPU calls"
            )

        if max_workers > 1:
            logger.info(
                f"⚡ PARALLEL EXECUTION: {max_workers} workers for test case verification"
            )
        else:
            logger.info(
                "🔧 SEQUENTIAL EXECUTION: Single-threaded test case verification"
            )

        # Provide diagnostic information if there are many failures
        if passed_count == 0 and total_problems > 0:
            logger.error("🚨 ALL EVALUATIONS FAILED - DIAGNOSTIC INFORMATION:")
            logger.error(
                f"   Generation failures: {generation_failures}/{total_problems}"
            )
            logger.error(
                f"   Execution failures: {execution_failures}/{total_problems}"
            )
            if generation_failures > 0:
                logger.error(
                    "   💡 Check your evaluation config - ensure temperature/do_sample compatibility"
                )
                logger.error(
                    f"   💡 Current config: temperature={self.config.temperature}, do_sample={getattr(self.config, 'do_sample', 'not set')}"
                )
        elif generation_failures > 0 or execution_failures > 0:
            logger.warning(
                f"📊 Failure breakdown: {generation_failures} generation failures, {execution_failures} execution failures"
            )

        summary_time = time.time() - summary_start_time
        logger.info(f"📋 Summary processing: {summary_time:.3f}s")

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
            dataset = load_dataset(
                "google-research-datasets/mbpp", "sanitized", cache_dir=save_dir
            )

            # Save as JSON for easy loading
            sanitized_path = os.path.join(save_dir, "sanitized-mbpp.json")
            with open(sanitized_path, "w") as f:
                json.dump(dataset["test"].to_list(), f, indent=2)

            logger.info(f"MBPP dataset saved to {sanitized_path}")
            return True

        except ImportError:
            logger.warning("datasets library not available, trying direct download...")

        # Fallback to direct download
        import urllib.request

        urls = [
            (
                "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
                "sanitized-mbpp.json",
            ),
            (
                "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
                "mbpp.jsonl",
            ),
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
        enabled=True, num_questions=5, dataset_path="./data/mbpp/sanitized-mbpp.json"
    )

    evaluator = MBPPEvaluator(config)

    if not evaluator.config.enabled:
        print("Dataset not found. Attempting to download...")
        if download_mbpp_dataset():
            print("Dataset downloaded successfully. Please run again.")
        else:
            print("Failed to download dataset. Please download manually.")
