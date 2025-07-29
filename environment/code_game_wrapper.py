"""Code generation game environment wrapper."""

import re
import io
import contextlib
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class CodeGameWrapper:
    """Environment wrapper for the code generation game."""
    
    def __init__(self, static_opponent_model=None, reward_shaper=None):
        """Initialize code game environment.
        
        Args:
            static_opponent_model: The static model that tries to guess outputs
            reward_shaper: Optional reward shaping function
        """
        self.static_opponent = static_opponent_model
        self.reward_shaper = reward_shaper
        self.current_episode = 0
        self.max_episodes = 1000
        
        # Game prompt template
        self.prompt_template = (
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
        
    def reset(self) -> Dict[str, Any]:
        """Reset environment for new episode."""
        self.current_episode += 1
        
        # Return the initial state (the prompt)
        return {
            'prompt': self.prompt_template,
            'episode': self.current_episode,
            'max_episodes': self.max_episodes
        }
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: The generated code response from the model
            
        Returns:
            next_state, reward, done, info
        """
        # Parse the generated code
        code, expected_output = self._parse_code_response(action)
        
        # Execute the code to get true output
        true_output = self._run_and_capture(code)
        
        # Get static opponent's prediction
        opponent_prediction = self._get_opponent_prediction(code)
        
        # Calculate reward
        reward = self._calculate_reward(opponent_prediction, true_output, expected_output)
        
        # Episode is done after one step
        done = True
        
        # Next state (not used since episode is done)
        next_state = {'prompt': '', 'episode': self.current_episode, 'max_episodes': self.max_episodes}
        
        # Info for logging
        info = {
            'code': code,
            'expected_output': expected_output,
            'true_output': true_output,
            'opponent_prediction': opponent_prediction,
            'raw_action': action
        }
        
        return next_state, reward, done, info
    
    def _parse_code_response(self, response: str) -> Tuple[str, str]:
        """Parse code and expected output from model response."""
        # Extract code
        code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        code = code_match.group(1).strip() if code_match else ""
        
        # Extract expected output
        output_match = re.search(r"```output\s*(.*?)```", response, re.DOTALL)
        expected_output = output_match.group(1).strip() if output_match else ""
        
        return code, expected_output
    
    def _run_and_capture(self, code: str) -> str:
        """Execute code and capture its stdout output."""
        if not code.strip():
            return "ERROR: No code provided"
            
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(code, {})  # empty global scope
        except Exception as e:
            return f"ERROR: {str(e)}"
        
        output = buffer.getvalue().strip()
        return output if output else "ERROR: No output"
    
    def _get_opponent_prediction(self, code: str) -> str:
        """Get prediction from static opponent model."""
        if self.static_opponent is None:
            return "ERROR: No static opponent"
            
        prompt = (
            "Examine this code and predict the integer output.\n"
            f"{code}\n\n"
            "Do not include any text, markdown, or explanation, just the number."
        )
        
        try:
            # Generate prediction
            response = self.static_opponent(
                prompt, 
                max_new_tokens=200, 
                do_sample=True, 
                temperature=0.7
            )[0]['generated_text']
            
            # Extract first number from response
            number_match = re.search(r"\b(-?\d+)\b", response)
            if number_match:
                return str(int(number_match.group(1)))
            else:
                return "ERROR: No number found"
                
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def _calculate_reward(self, opponent_prediction: str, true_output: str, expected_output: str) -> float:
        """Calculate reward based on game outcome."""
        # Base reward structure
        reward = 0.0
        
        # Check if code executed successfully
        if true_output.startswith("ERROR:"):
            reward -= 10.0  # Penalty for broken code
            return reward
            
        # Check if expected output matches true output (code correctness)
        if expected_output != true_output:
            reward -= 5.0  # Penalty for incorrect expected output
            return reward
            
        # Check if opponent prediction matches true output
        if opponent_prediction.startswith("ERROR:"):
            reward += 1.0  # Small reward if opponent fails
        elif opponent_prediction == true_output:
            reward -= 1.0  # Penalty if opponent guesses correctly
        else:
            reward += 10.0  # Large reward if opponent guesses incorrectly
            
        # Apply reward shaping if available
        if self.reward_shaper:
            reward = self.reward_shaper.compute_reward(
                code=true_output,
                opponent_correct=(opponent_prediction == true_output),
                base_reward=reward
            )
            
        return reward
    
    def get_action_space_size(self) -> int:
        """Return action space size (not applicable for text generation)."""
        return -1  # Continuous action space (text generation)
    
    def get_observation_space_size(self) -> int:
        """Return observation space size (not applicable for text input)."""
        return -1  # Variable length text input