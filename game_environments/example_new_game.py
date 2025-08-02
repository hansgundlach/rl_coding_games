#!/usr/bin/env python3
"""
Example implementation of a new game for the extensible framework.

This demonstrates how to create a new competitive game where LLMs submit code strategies.
In this example, players submit sorting algorithms and compete on speed/correctness.
"""

import re
import random
import time
from typing import Dict, List, Any, Tuple

from .base_game import BaseGame, PlayerSubmission, GameResult


class SortingChallenge(BaseGame):
    """
    Example game: Sorting Algorithm Challenge
    
    Players submit sorting algorithm code and compete on correctness and speed.
    This demonstrates the extensible framework pattern.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.test_cases = config.get("test_cases", [
            [3, 1, 4, 1, 5, 9, 2, 6],
            [100, 50, 25, 75],
            list(range(100, 0, -1)),  # Reverse sorted
            [1, 1, 1, 1, 1],  # All same
            []  # Empty list
        ])
        
    def get_game_name(self) -> str:
        return "Sorting Algorithm Challenge"
        
    def get_player_prompt(self, player_id: int, role: str) -> str:
        """Generate a prompt for players to write their sorting algorithm."""
        return f"""You are Player {player_id} in a Sorting Algorithm Challenge. Your goal is to write a sorting function that is both correct and efficient.

## Your Task:
Write a Python function called `sort_algorithm` that takes a list of numbers and returns a sorted list.

```python
def sort_algorithm(numbers):
    \"\"\"
    Sort the given list of numbers in ascending order.
    
    Args:
        numbers: List of integers/floats to sort
        
    Returns:
        List of numbers sorted in ascending order
    \"\"\"
    # Your sorting algorithm here
    # You can implement any sorting algorithm: bubble sort, merge sort, quick sort, etc.
    # Focus on correctness first, then efficiency
    
    return sorted(numbers)  # Replace with your implementation
```

## Scoring:
- Correctness: Your algorithm must produce the correct sorted output
- Speed: Faster algorithms get bonus points
- Code execution failures result in -1 reward

Write an efficient and correct sorting algorithm:"""

    def extract_code_from_response(self, response: str) -> str:
        """Extract the sorting function from the LLM response."""
        # Look for code in markdown format
        code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Fallback: look for function definition
        func_match = re.search(r"def sort_algorithm\(.*?\):(.*?)(?=\n\S|\n*$)", response, re.DOTALL)
        if func_match:
            return f"def sort_algorithm{func_match.group(0)[len('def sort_algorithm'):]}"
        
        # Last resort: return the response as-is
        return response.strip()
        
    def validate_submission(self, code: str, player_id: int, role: str) -> Tuple[bool, str]:
        """Validate that the code contains a proper sort_algorithm function."""
        # Check if function is defined
        if "def sort_algorithm(" not in code:
            return False, "Code must contain a 'sort_algorithm' function"
            
        # Try to compile the code
        try:
            compile(code, f"<player_{player_id}_sort>", "exec")
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Compilation error: {str(e)}"
            
        return True, ""
        
    def get_expected_function_name(self) -> str:
        return "sort_algorithm"
        
    def play_game(self, submissions: List[PlayerSubmission]) -> GameResult:
        """
        Run a sorting challenge between the submitted algorithms.
        """
        if len(submissions) < 2:
            return GameResult(
                player_rewards={i: self.get_reward_on_execution_failure() for i in range(len(submissions))},
                game_data={"error": f"Need at least 2 players, got {len(submissions)}"},
                execution_logs=["Insufficient players"],
                successful_submissions=0
            )
        
        # Test each submission on all test cases
        player_results = {}
        successful_count = 0
        execution_logs = []
        
        for submission in submissions:
            if not submission.compilation_success:
                player_results[submission.player_id] = {
                    "total_time": float('inf'),
                    "correct_count": 0,
                    "failed": True,
                    "error": submission.compilation_error
                }
                continue
                
            successful_count += 1
            total_time = 0.0
            correct_count = 0
            failed = False
            
            for i, test_case in enumerate(self.test_cases):
                # Execute the sorting algorithm
                start_time = time.time()
                result = self.code_executor.execute_strategy_code(
                    submission.extracted_code,
                    "sort_algorithm",
                    test_case.copy()
                )
                end_time = time.time()
                
                if result["success"]:
                    # Check correctness
                    try:
                        output = result["result"]
                        expected = sorted(test_case)
                        if output == expected:
                            correct_count += 1
                            total_time += (end_time - start_time)
                        else:
                            execution_logs.append(f"Player {submission.player_id}, Test {i}: Incorrect output")
                    except:
                        execution_logs.append(f"Player {submission.player_id}, Test {i}: Invalid output format")
                else:
                    failed = True
                    execution_logs.append(f"Player {submission.player_id}, Test {i}: Execution failed - {result['error']}")
                    break
            
            player_results[submission.player_id] = {
                "total_time": total_time,
                "correct_count": correct_count,
                "failed": failed
            }
        
        # Determine winner and assign rewards
        rewards = {}
        best_player = None
        best_score = -1
        
        for submission in submissions:
            player_id = submission.player_id
            result = player_results[player_id]
            
            if result.get("failed", True) or result["correct_count"] == 0:
                # Failed or got no test cases correct
                rewards[player_id] = self.get_reward_on_execution_failure()
            else:
                # Score based on correctness and speed
                correctness_score = result["correct_count"] / len(self.test_cases)
                speed_score = 1.0 / (result["total_time"] + 0.001)  # Avoid division by zero
                combined_score = correctness_score * 0.7 + speed_score * 0.3
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_player = player_id
        
        # Assign rewards: winner gets +1, others get -1
        for submission in submissions:
            if submission.player_id == best_player:
                rewards[submission.player_id] = 1.0
            else:
                rewards[submission.player_id] = rewards.get(submission.player_id, -1.0)
        
        execution_logs.append(f"Winner: Player {best_player} with score {best_score:.4f}")
        
        return GameResult(
            player_rewards=rewards,
            game_data={
                "winner": best_player,
                "player_results": player_results,
                "test_cases_count": len(self.test_cases)
            },
            execution_logs=execution_logs,
            successful_submissions=successful_count
        )