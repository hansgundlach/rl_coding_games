#!/usr/bin/env python3
"""
Iterated Prisoner's Dilemma Game Environment

Players submit code strategies that compete in an iterated prisoner's dilemma tournament.
Each strategy is a Python function that receives the history of previous rounds and returns
either 'cooperate' or 'defect'.
"""

import re
import random
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from .base_game import BaseGame, PlayerSubmission, GameResult


@dataclass
class RoundResult:
    """Result of a single round in the prisoner's dilemma."""
    player1_action: str
    player2_action: str
    player1_payoff: float
    player2_payoff: float
    round_number: int


class IteratedPrisonersDilemma(BaseGame):
    """
    Iterated Prisoner's Dilemma game where players submit strategy code.
    
    Players compete by writing functions that decide whether to cooperate or defect
    based on the history of previous rounds.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_rounds = config.get("num_rounds", 100)
        self.payoff_matrix = config.get("payoff_matrix", {
            ("cooperate", "cooperate"): (3, 3),
            ("cooperate", "defect"): (0, 5),
            ("defect", "cooperate"): (5, 0),
            ("defect", "defect"): (1, 1),
        })
        
    def get_game_name(self) -> str:
        return "Iterated Prisoner's Dilemma"
        
    def get_player_prompt(self, player_id: int, role: str) -> str:
        """Generate a prompt for players to write their prisoner's dilemma strategy."""
        return f"""You are Player {player_id} in an Iterated Prisoner's Dilemma tournament. Your goal is to write a strategy function that will compete against another player's strategy over {self.num_rounds} rounds.

## Game Rules:
In each round, you can either 'cooperate' or 'defect'. The payoffs are:
- Both cooperate: Both get 3 points
- Both defect: Both get 1 point  
- One cooperates, one defects: Cooperator gets 0, defector gets 5

## Your Task:
Write a Python function called `strategy` that takes the game history and returns your action.

```python
def strategy(my_history, opponent_history, round_number):
    \"\"\"
    Your prisoner's dilemma strategy.
    
    Args:
        my_history: List of your previous actions ['cooperate', 'defect', ...]
        opponent_history: List of opponent's previous actions ['cooperate', 'defect', ...]
        round_number: Current round number (1-based)
        
    Returns:
        str: Either 'cooperate' or 'defect'
    \"\"\"
    # Your strategy logic here
    # Examples:
    # - Always cooperate: return 'cooperate'
    # - Always defect: return 'defect'
    # - Tit-for-tat: cooperate first, then copy opponent's last move
    # - More complex strategies based on patterns, forgiveness, etc.
    
    return 'cooperate'  # Replace with your strategy
```

## Strategy Ideas:
- **Tit-for-Tat**: Cooperate first, then copy opponent's previous move
- **Generous Tit-for-Tat**: Like tit-for-tat but occasionally forgive defections
- **Pavlov (Win-Stay, Lose-Shift)**: Repeat if you got high payoff, switch if low
- **Random**: Cooperate or defect randomly
- **Grudger**: Cooperate until opponent defects, then always defect
- **Pattern Recognition**: Try to detect and exploit opponent patterns

Write a strategy that maximizes your total score over {self.num_rounds} rounds:"""

    def extract_code_from_response(self, response: str) -> str:
        """Extract the strategy function from the LLM response."""
        # Look for code in markdown format
        code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Fallback: look for function definition
        func_match = re.search(r"def strategy\(.*?\):(.*?)(?=\n\S|\n*$)", response, re.DOTALL)
        if func_match:
            return f"def strategy{func_match.group(0)[len('def strategy'):]}"
        
        # Last resort: return the response as-is
        return response.strip()
        
    def validate_submission(self, code: str, player_id: int, role: str) -> Tuple[bool, str]:
        """Validate that the code contains a proper strategy function."""
        # Check if function is defined
        if "def strategy(" not in code:
            return False, "Code must contain a 'strategy' function"
            
        # Try to compile the code
        try:
            compile(code, f"<player_{player_id}_strategy>", "exec")
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Compilation error: {str(e)}"
            
        return True, ""
        
    def get_expected_function_name(self) -> str:
        return "strategy"
        
    def get_max_players(self) -> int:
        return 2
        
    def play_game(self, submissions: List[PlayerSubmission]) -> GameResult:
        """
        Run a complete iterated prisoner's dilemma game.
        
        The game consists of multiple rounds where each player's strategy function
        is called to get their action, then payoffs are calculated.
        """
        if len(submissions) != 2:
            return GameResult(
                player_rewards={i: self.get_reward_on_execution_failure() for i in range(len(submissions))},
                game_data={"error": f"Expected 2 players, got {len(submissions)}"},
                execution_logs=["Invalid number of players"],
                successful_submissions=0
            )
            
        # Check if both submissions compiled successfully
        successful_count = sum(1 for sub in submissions if sub.compilation_success)
        if successful_count < 2:
            # If either player has compilation errors, both get failure reward
            rewards = {}
            for sub in submissions:
                if sub.compilation_success:
                    rewards[sub.player_id] = self.get_reward_on_execution_failure()  # Penalty for opponent's failure
                else:
                    rewards[sub.player_id] = self.get_reward_on_execution_failure()
                    
            return GameResult(
                player_rewards=rewards,
                game_data={"error": "One or more players had compilation errors"},
                execution_logs=[f"Player {sub.player_id}: {'OK' if sub.compilation_success else sub.compilation_error}" for sub in submissions],
                successful_submissions=successful_count
            )
        
        # Initialize game state
        player1, player2 = submissions[0], submissions[1]
        history = {
            player1.player_id: [],
            player2.player_id: []
        }
        total_payoffs = {
            player1.player_id: 0.0,
            player2.player_id: 0.0
        }
        round_results = []
        execution_logs = []
        
        # Play the iterated game
        for round_num in range(1, self.num_rounds + 1):
            # Get actions from both players
            p1_action = self._get_player_action(
                player1, 
                history[player1.player_id], 
                history[player2.player_id], 
                round_num
            )
            
            p2_action = self._get_player_action(
                player2, 
                history[player2.player_id], 
                history[player1.player_id], 
                round_num
            )
            
            # Handle execution failures
            if p1_action is None or p2_action is None:
                # If either player fails, end the game and give failure rewards
                execution_logs.append(f"Round {round_num}: Strategy execution failed")
                return GameResult(
                    player_rewards={
                        player1.player_id: self.get_reward_on_execution_failure(),
                        player2.player_id: self.get_reward_on_execution_failure()
                    },
                    game_data={
                        "rounds_completed": round_num - 1,
                        "failure_reason": "Strategy execution failed",
                        "round_results": round_results
                    },
                    execution_logs=execution_logs,
                    successful_submissions=successful_count
                )
            
            # Calculate payoffs for this round
            payoff_key = (p1_action, p2_action)
            if payoff_key in self.payoff_matrix:
                p1_payoff, p2_payoff = self.payoff_matrix[payoff_key]
            else:
                # Invalid actions default to defect vs defect
                p1_payoff, p2_payoff = self.payoff_matrix[("defect", "defect")]
                execution_logs.append(f"Round {round_num}: Invalid actions, defaulting to defect vs defect")
            
            # Update totals and history
            total_payoffs[player1.player_id] += p1_payoff
            total_payoffs[player2.player_id] += p2_payoff
            
            history[player1.player_id].append(p1_action)
            history[player2.player_id].append(p2_action)
            
            # Record round result
            round_result = RoundResult(
                player1_action=p1_action,
                player2_action=p2_action,
                player1_payoff=p1_payoff,
                player2_payoff=p2_payoff,
                round_number=round_num
            )
            round_results.append(round_result)
            
            execution_logs.append(f"Round {round_num}: P1={p1_action}, P2={p2_action}, Payoffs=({p1_payoff}, {p2_payoff})")
        
        # Determine winner and assign rewards
        p1_total = total_payoffs[player1.player_id]
        p2_total = total_payoffs[player2.player_id]
        
        if p1_total > p2_total:
            # Player 1 wins
            rewards = {player1.player_id: 1.0, player2.player_id: -1.0}
        elif p2_total > p1_total:
            # Player 2 wins
            rewards = {player1.player_id: -1.0, player2.player_id: 1.0}
        else:
            # Tie
            rewards = {player1.player_id: 0.0, player2.player_id: 0.0}
        
        execution_logs.append(f"Final scores: P1={p1_total}, P2={p2_total}")
        execution_logs.append(f"Winner: {'P1' if p1_total > p2_total else 'P2' if p2_total > p1_total else 'Tie'}")
        
        return GameResult(
            player_rewards=rewards,
            game_data={
                "rounds_completed": self.num_rounds,
                "final_payoffs": total_payoffs,
                "round_results": round_results,
                "winner": "player1" if p1_total > p2_total else "player2" if p2_total > p1_total else "tie"
            },
            execution_logs=execution_logs,
            successful_submissions=successful_count
        )
    
    def _get_player_action(self, player: PlayerSubmission, my_history: List[str], 
                          opponent_history: List[str], round_number: int) -> str:
        """Get a player's action for the current round."""
        try:
            # Execute the player's strategy function
            result = self.code_executor.execute_strategy_code(
                player.extracted_code,
                "strategy", 
                my_history.copy(),  # Pass copies to avoid modification
                opponent_history.copy(),
                round_number
            )
            
            if result["success"] and result["result"]:
                action = str(result["result"]).lower().strip()
                # Normalize action
                if action in ["cooperate", "c", "coop"]:
                    return "cooperate"
                elif action in ["defect", "d", "def"]:
                    return "defect"
                else:
                    # Invalid action defaults to defect
                    return "defect"
            else:
                # Execution failed
                return None
                
        except Exception as e:
            # Execution error
            return None