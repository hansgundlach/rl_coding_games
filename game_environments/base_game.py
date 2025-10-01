#!/usr/bin/env python3
"""
Base Game Environment Framework for SPIRAL Self-Play Training

This provides an extensible framework for implementing different game environments
where LLMs compete by submitting code strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import subprocess
import tempfile
import time
import sys
import os


@dataclass
class PlayerSubmission:
    """Represents a player's code submission for a game."""
    player_id: int
    role: str
    prompt: str
    response: str
    extracted_code: str
    compilation_success: bool
    compilation_error: str = ""


@dataclass
class GameResult:
    """Results from a complete game between players."""
    player_rewards: Dict[int, float]  # player_id -> reward
    game_data: Dict[str, Any]  # Game-specific data
    execution_logs: List[str]  # Execution logs
    successful_submissions: int  # Number of successfully compiled submissions


class CodeExecutor:
    """Safe code execution environment for player strategies."""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
    
    def execute_strategy_code(self, code: str, function_name: str, *args) -> Dict:
        """
        Execute a player's strategy code safely.
        
        Args:
            code: The player's submitted code
            function_name: Name of the function to call
            *args: Arguments to pass to the function
            
        Returns:
            dict: {
                'success': bool,
                'result': Any,
                'error': str,
                'execution_time': float,
                'timeout': bool
            }
        """
        result = {
            "success": False,
            "result": None,
            "error": "",
            "execution_time": 0.0,
            "timeout": False,
        }
        
        # Create wrapper code that calls the function
        wrapper_code = f"""
import sys
import json

# Player's submitted code
{code}

# Call the required function and return result
try:
    if callable(globals().get('{function_name}')):
        result = {function_name}({', '.join(repr(arg) for arg in args)})
        print(json.dumps({{"success": True, "result": result}}))
    else:
        print(json.dumps({{"success": False, "error": f"Function '{function_name}' not found or not callable"}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
"""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapper_code)
            temp_file = f.name
        
        try:
            start_time = time.time()
            
            # Execute code in subprocess with timeout
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tempfile.gettempdir(),
                env={"PYTHONPATH": ""},  # Minimal environment
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                result["execution_time"] = time.time() - start_time
                
                if process.returncode == 0 and stdout.strip():
                    try:
                        # Parse the JSON result
                        import json
                        output = json.loads(stdout.strip())
                        result["success"] = output.get("success", False)
                        result["result"] = output.get("result")
                        result["error"] = output.get("error", "")
                    except json.JSONDecodeError:
                        result["error"] = f"Invalid JSON output: {stdout[:100]}"
                else:
                    result["error"] = stderr.strip() or f"Process failed with code {process.returncode}"
                    
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                result["timeout"] = True
                result["error"] = f"Code execution timed out after {self.timeout} seconds"
                result["execution_time"] = self.timeout
                
        except Exception as e:
            result["error"] = f"Execution error: {str(e)}"
            result["execution_time"] = time.time() - start_time
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
                
        return result


class BaseGame(ABC):
    """
    Abstract base class for game environments.
    
    Games should inherit from this class and implement the required methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.code_executor = CodeExecutor(timeout=config.get("timeout", 5))
        
    @abstractmethod
    def get_game_name(self) -> str:
        """Return the name of this game."""
        pass
        
    @abstractmethod
    def get_player_prompt(self, player_id: int, role: str) -> str:
        """
        Generate a prompt for a player to write their strategy code.
        
        Args:
            player_id: Unique identifier for the player
            role: Player's role in the game (if applicable)
            
        Returns:
            str: Prompt text for the LLM
        """
        pass
        
    @abstractmethod
    def extract_code_from_response(self, response: str) -> str:
        """
        Extract the strategy code from the LLM's response.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            str: Extracted code
        """
        pass
        
    @abstractmethod
    def validate_submission(self, code: str, player_id: int, role: str) -> Tuple[bool, str]:
        """
        Validate a player's code submission.
        
        Args:
            code: The extracted code
            player_id: Player identifier
            role: Player role
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        pass
        
    @abstractmethod
    def play_game(self, submissions: List[PlayerSubmission]) -> GameResult:
        """
        Run a complete game with the given player submissions.
        
        Args:
            submissions: List of player code submissions
            
        Returns:
            GameResult: Complete results of the game
        """
        pass
        
    def get_reward_on_execution_failure(self) -> float:
        """Return the reward given when a player's code fails to execute."""
        return -1.0
        
    def get_expected_function_name(self) -> str:
        """Return the name of the function players should implement."""
        return "strategy"
        
    def get_max_players(self) -> int:
        """Return the maximum number of players for this game."""
        return 2
        
    def get_roles(self) -> List[str]:
        """Return the list of possible player roles."""
        return ["player"]