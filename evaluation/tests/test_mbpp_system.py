#!/usr/bin/env python3
"""
Comprehensive test suite for MBPP evaluation system.
Tests all components with sample data and mock models.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import components to test
from evaluation.mbpp.evaluator import MBPPEvaluator, EvalConfig
from evaluation.configs.loader import load_eval_config, create_eval_config_for_training
from evaluation.datasets import load_sample_dataset, download_mbpp_dataset


class TestEvalConfig(unittest.TestCase):
    """Test EvalConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EvalConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.num_questions, 10)
        self.assertTrue(config.eval_at_start)
        self.assertTrue(config.eval_at_end)
        self.assertIsNone(config.eval_interval_steps)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EvalConfig(
            enabled=False,
            num_questions=5,
            temperature=0.1
        )
        self.assertFalse(config.enabled)
        self.assertEqual(config.num_questions, 5)
        self.assertEqual(config.temperature, 0.1)


class TestMBPPEvaluator(unittest.TestCase):
    """Test MBPPEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temp directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample dataset
        self.sample_data = [
            {
                "task_id": 1,
                "prompt": "Write a function that adds two numbers.",
                "code": "def add(a, b):\n    return a + b",
                "test_list": [
                    "assert add(2, 3) == 5",
                    "assert add(0, 0) == 0"
                ]
            }
        ]
        
        self.dataset_path = os.path.join(self.temp_dir, "test_mbpp.json")
        with open(self.dataset_path, 'w') as f:
            json.dump(self.sample_data, f)
        
        # Create test config
        self.config = EvalConfig(
            enabled=True,
            num_questions=1,
            dataset_path=self.dataset_path,
            save_results=False,
            verbose=False
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = MBPPEvaluator(self.config)
        self.assertTrue(evaluator.config.enabled)
        self.assertEqual(len(evaluator.problems), 1)
        self.assertEqual(evaluator.problems[0]["task_id"], 1)
    
    def test_disabled_evaluator(self):
        """Test disabled evaluator."""
        config = EvalConfig(enabled=False)
        evaluator = MBPPEvaluator(config)
        self.assertFalse(evaluator.config.enabled)
    
    def test_should_evaluate(self):
        """Test should_evaluate logic."""
        evaluator = MBPPEvaluator(self.config)
        
        # Test start/end conditions
        self.assertTrue(evaluator.should_evaluate(is_start=True))
        self.assertTrue(evaluator.should_evaluate(is_end=True))
        self.assertFalse(evaluator.should_evaluate())
        
        # Test interval-based evaluation
        evaluator.config.eval_interval_steps = 2
        evaluator.step_count = 2
        self.assertTrue(evaluator.should_evaluate())
        
        evaluator.step_count = 3
        self.assertFalse(evaluator.should_evaluate())
    
    def test_create_prompt(self):
        """Test prompt creation."""
        evaluator = MBPPEvaluator(self.config)
        problem = self.sample_data[0]
        
        prompt = evaluator.create_prompt(problem)
        
        self.assertIn("Write a function that adds two numbers", prompt)
        self.assertIn("assert add(2, 3) == 5", prompt)
        self.assertTrue(prompt.startswith('"""'))
    
    def test_extract_code(self):
        """Test code extraction from completions."""
        evaluator = MBPPEvaluator(self.config)
        
        # Test markdown format
        completion_markdown = "```python\ndef test():\n    return 42\n```"
        code = evaluator.extract_code(completion_markdown)
        self.assertEqual(code, "def test():\n    return 42")
        
        # Test plain format
        completion_plain = "def test():\n    return 42"
        code = evaluator.extract_code(completion_plain)
        self.assertEqual(code, "def test():\n    return 42")
    
    def test_safe_execute_code(self):
        """Test safe code execution."""
        evaluator = MBPPEvaluator(self.config)
        
        # Test successful execution
        good_code = "def add(a, b):\n    return a + b"
        test_cases = ["assert add(2, 3) == 5"]
        result = evaluator.safe_execute_code(good_code, test_cases)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['error'], '')
        self.assertFalse(result['timeout'])
        
        # Test failed execution
        bad_code = "def broken():\n    return undefined_variable"
        test_cases = ["assert broken() == 1"]
        result = evaluator.safe_execute_code(bad_code, test_cases)
        
        self.assertFalse(result['success'])
        self.assertIn('NameError', result['error'])
        self.assertFalse(result['timeout'])
    
    def test_get_eval_problems(self):
        """Test problem selection."""
        evaluator = MBPPEvaluator(self.config)
        problems = evaluator.get_eval_problems()
        
        self.assertEqual(len(problems), 1)  # num_questions = 1
        self.assertEqual(problems[0]['task_id'], 1)


class TestConfigLoader(unittest.TestCase):
    """Test configuration loading system."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_eval_config()
        self.assertIsInstance(config, EvalConfig)
        self.assertTrue(config.enabled)
    
    def test_config_with_overrides(self):
        """Test configuration with overrides."""
        config = load_eval_config(num_questions=5, temperature=0.1)
        self.assertEqual(config.num_questions, 5)
        self.assertEqual(config.temperature, 0.1)
    
    def test_training_specific_config(self):
        """Test training-specific configuration."""
        config = create_eval_config_for_training("grpo_code_game")
        self.assertIn("grpo_code_game", config.results_dir)
        
        config = create_eval_config_for_training("grpo_code_execution")
        self.assertIn("grpo_code_execution", config.results_dir)
    
    @patch.dict(os.environ, {'MBPP_EVAL_NUM_QUESTIONS': '15'})
    def test_env_override(self):
        """Test environment variable override."""
        config = load_eval_config()
        self.assertEqual(config.num_questions, 15)


class TestDatasetUtils(unittest.TestCase):
    """Test dataset utilities."""
    
    def test_load_sample_dataset(self):
        """Test loading sample dataset."""
        # This will fail if sample dataset doesn't exist, which is ok for testing
        data = load_sample_dataset()
        # Should return empty list if file doesn't exist
        self.assertIsInstance(data, list)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a more comprehensive sample dataset
        self.sample_data = [
            {
                "task_id": 11,
                "prompt": "Write a function to add two numbers.",
                "code": "def add(a, b):\n    return a + b",
                "test_list": [
                    "assert add(2, 3) == 5",
                    "assert add(-1, 1) == 0",
                    "assert add(0, 0) == 0"
                ]
            },
            {
                "task_id": 12,
                "prompt": "Write a function to multiply two numbers.",
                "code": "def multiply(a, b):\n    return a * b",
                "test_list": [
                    "assert multiply(2, 3) == 6",
                    "assert multiply(0, 5) == 0",
                    "assert multiply(-2, 3) == -6"
                ]
            }
        ]
        
        self.dataset_path = os.path.join(self.temp_dir, "integration_test.json")
        with open(self.dataset_path, 'w') as f:
            json.dump(self.sample_data, f)
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow without real model."""
        # Create config
        config = EvalConfig(
            enabled=True,
            num_questions=2,
            dataset_path=self.dataset_path,
            save_results=False,
            verbose=False
        )
        
        # Initialize evaluator
        evaluator = MBPPEvaluator(config)
        self.assertTrue(evaluator.config.enabled)
        self.assertEqual(len(evaluator.problems), 2)
        
        # Test evaluation control
        self.assertTrue(evaluator.should_evaluate(is_start=True))
        
        # Test problem selection
        problems = evaluator.get_eval_problems()
        self.assertLessEqual(len(problems), 2)
        
        # Test prompt creation
        for problem in problems:
            prompt = evaluator.create_prompt(problem)
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 0)
    
    def test_mock_model_evaluation(self):
        """Test evaluation with mock model."""
        config = EvalConfig(
            enabled=True,
            num_questions=1,
            dataset_path=self.dataset_path,
            save_results=False,
            verbose=False
        )
        
        evaluator = MBPPEvaluator(config)
        
        # Create mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Configure mocks
        mock_tokenizer.return_value = {
            'input_ids': Mock(shape=[1, 10])
        }
        mock_tokenizer.decode.return_value = "def add(a, b):\n    return a + b"
        
        mock_model.device = "cpu"
        mock_model.generate.return_value = [Mock()]
        
        # Mock torch.cuda
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.no_grad'):
                # Test single problem evaluation
                problem = evaluator.problems[0]
                result = evaluator.evaluate_single_problem(problem, mock_model, mock_tokenizer)
                
                self.assertIn('task_id', result)
                self.assertIn('passed', result)
                self.assertIsInstance(result['passed'], bool)


def run_tests():
    """Run all tests."""
    print("🧪 Running MBPP Evaluation System Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestEvalConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestMBPPEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
        
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)