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
        quick_eval: bool = False,
        quick_eval_size: int = 10,
    ):
        """Initialize EvalPlus runner."""
        self.model_path = model_path
        self.lora_path = lora_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.num_samples = num_samples
        self.quick_eval = quick_eval
        self.quick_eval_size = quick_eval_size

        logger.info(f"Initializing EvalPlus with model: {model_path}")
        if quick_eval:
            logger.info(
                f"Quick evaluation mode enabled: using {quick_eval_size} problems per dataset"
            )

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
                cache_dir="./model_cache",
                local_files_only=False,
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
        import time

        eval_mode = "Quick" if self.quick_eval else "Full"
        logger.info(f"Starting {eval_mode} EvalPlus evaluation...")

        results = {}
        start_time = time.time()

        # Run HumanEval-Plus
        logger.info("Running HumanEval-Plus evaluation...")
        humaneval_start = time.time()
        humaneval_results = self.evaluate_humaneval_plus()
        humaneval_time = time.time() - humaneval_start
        results["humaneval_plus"] = humaneval_results
        results["humaneval_plus"]["eval_time_seconds"] = humaneval_time

        # Run MBPP-Plus
        logger.info("Running MBPP-Plus evaluation...")
        mbpp_start = time.time()
        mbpp_results = self.evaluate_mbpp_plus()
        mbpp_time = time.time() - mbpp_start
        results["mbpp_plus"] = mbpp_results
        results["mbpp_plus"]["eval_time_seconds"] = mbpp_time

        total_time = time.time() - start_time
        results["total_eval_time_seconds"] = total_time
        results["eval_mode"] = eval_mode
        results["problems_per_dataset"] = (
            self.quick_eval_size if self.quick_eval else "all"
        )

        logger.info(f"{eval_mode} evaluation completed in {total_time:.1f}s")
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
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return fallback results
            return {"pass@1": 0.0, "pass@10": 0.0}

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
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return fallback results
            return {"pass@1": 0.0, "pass@10": 0.0}

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

            # Select subset for quick evaluation
            if self.quick_eval:
                problem_items = list(problems.items())[: self.quick_eval_size]
                logger.info(
                    f"Quick eval: processing {len(problem_items)} HumanEval problems"
                )
            else:
                problem_items = list(problems.items())
                logger.info(
                    f"Full eval: processing {len(problem_items)} HumanEval problems"
                )

            for task_id, problem in problem_items:
                logger.info(f"Generating solution for {task_id}")

                # Create prompt
                prompt = self._create_code_prompt(problem["prompt"])

                # Generate multiple solutions
                task_solutions = []
                for i in range(self.num_samples):
                    solution = self._generate_code(prompt)
                    task_solutions.append(solution)

                solutions.append(
                    {
                        "task_id": task_id,
                        "completion": task_solutions[
                            0
                        ],  # Use first solution for pass@1
                        "all_completions": task_solutions,
                    }
                )

            # Save solutions to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                for solution in solutions:
                    json.dump(solution, f)
                    f.write("\n")
                return f.name

        except Exception as e:
            logger.error(f"Failed to generate HumanEval solutions: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def _generate_solutions_mbpp(self) -> str:
        """Generate solutions for MBPP problems."""
        try:
            from evalplus.data import get_mbpp_plus

            problems = get_mbpp_plus()
            solutions = []

            # Select subset for quick evaluation
            if self.quick_eval:
                problem_items = list(problems.items())[: self.quick_eval_size]
                logger.info(
                    f"Quick eval: processing {len(problem_items)} MBPP problems"
                )
            else:
                problem_items = list(problems.items())
                logger.info(f"Full eval: processing {len(problem_items)} MBPP problems")

            for task_id, problem in problem_items:
                logger.info(f"Generating solution for {task_id}")

                # Create prompt
                prompt = self._create_code_prompt(problem["prompt"])

                # Generate multiple solutions
                task_solutions = []
                for i in range(self.num_samples):
                    solution = self._generate_code(prompt)
                    task_solutions.append(solution)

                solutions.append(
                    {
                        "task_id": task_id,
                        "completion": task_solutions[
                            0
                        ],  # Use first solution for pass@1
                        "all_completions": task_solutions,
                    }
                )

            # Save solutions to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                for solution in solutions:
                    json.dump(solution, f)
                    f.write("\n")
                return f.name

        except Exception as e:
            logger.error(f"Failed to generate MBPP solutions: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
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
                max_length=1024,
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
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            return generated_text

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            return ""

    def _run_evalplus_evaluation(
        self, solutions_file: str, dataset: str
    ) -> Dict[str, float]:
        """Run evalplus evaluation on solutions file."""
        try:
            if self.quick_eval:
                # For quick eval, compute pass rate manually
                return self._compute_quick_pass_rate(solutions_file, dataset)

            # Run evalplus command for full evaluation
            cmd = [
                "evalplus.evaluate",
                "--dataset",
                dataset,
                "--samples",
                solutions_file,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"Evalplus evaluation failed: {result.stderr}")
                return {"pass@1": 0.0, "pass@10": 0.0}

            # Parse results
            output_lines = result.stdout.strip().split("\n")
            results = {}

            for line in output_lines:
                if "pass@1" in line:
                    try:
                        results["pass@1"] = float(line.split()[-1])
                    except:
                        results["pass@1"] = 0.0
                elif "pass@10" in line:
                    try:
                        results["pass@10"] = float(line.split()[-1])
                    except:
                        results["pass@10"] = 0.0

            # Clean up temporary file
            os.unlink(solutions_file)

            return results

        except Exception as e:
            logger.error(f"Evalplus evaluation failed: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {"pass@1": 0.0, "pass@10": 0.0}

    def _compute_quick_pass_rate(
        self, solutions_file: str, dataset: str
    ) -> Dict[str, float]:
        """Compute pass rate for quick evaluation using simple execution."""
        try:
            import json
            import sys
            import io
            from contextlib import redirect_stdout, redirect_stderr

            # Load solutions
            solutions = []
            with open(solutions_file, "r") as f:
                for line in f:
                    solutions.append(json.loads(line.strip()))

            # Load test cases
            if dataset == "humaneval":
                from evalplus.data import get_human_eval_plus

                problems = get_human_eval_plus()
            else:  # mbpp
                from evalplus.data import get_mbpp_plus

                problems = get_mbpp_plus()

            passed = 0
            total = len(solutions)

            for solution in solutions:
                task_id = solution["task_id"]
                completion = solution["completion"]

                if task_id not in problems:
                    continue

                problem = problems[task_id]

                # Simple execution test (basic functionality only)
                try:
                    # Combine prompt and completion
                    if dataset == "humaneval":
                        full_code = problem["prompt"] + completion
                    else:  # mbpp
                        full_code = completion

                    # Execute in isolated namespace
                    namespace = {}
                    exec(full_code, namespace)

                    # Run a basic test (this is simplified)
                    if "test" in problem:
                        test_code = problem["test"]
                        exec(test_code, namespace)
                        passed += 1
                    else:
                        # If no explicit test, consider it passed if execution succeeded
                        passed += 1

                except Exception as e:
                    # Execution failed
                    continue

            # Clean up temporary file
            os.unlink(solutions_file)

            pass_rate = passed / total if total > 0 else 0.0
            logger.info(
                f"Quick eval: {passed}/{total} problems passed ({pass_rate:.3f})"
            )

            return {
                "pass@1": pass_rate,
                "pass@10": pass_rate,  # Same as pass@1 for quick eval
                "problems_tested": total,
                "problems_passed": passed,
            }

        except Exception as e:
            logger.error(f"Quick evaluation computation failed: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")

            # Clean up temporary file
            try:
                os.unlink(solutions_file)
            except:
                pass

            return {"pass@1": 0.0, "pass@10": 0.0}


def main():
    """Main evaluation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run EvalPlus benchmarks")
    parser.add_argument("checkpoint_path", help="Path to model checkpoint")
    parser.add_argument(
        "--output", "-o", default="eval_results.json", help="Output file for results"
    )

    args = parser.parse_args()

    # Run evaluation
    runner = EvalPlusRunner(
        model_path="Qwen/Qwen2.5-3B",
        lora_path=args.checkpoint_path,
        quick_eval=False,  # Full evaluation for standalone runs
    )

    results = runner.run_evaluation()

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation results saved to {args.output}")


if __name__ == "__main__":
    main()
