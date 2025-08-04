#!/usr/bin/env python3
"""
Iterated Prisoner's Dilemma Game Environment

Players submit code strategies that compete in an iterated prisoner's dilemma tournament.
Each strategy is a Python function that receives the history of previous rounds and returns
either 'cooperate' or 'defect'.
"""

import re
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
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
    noise_applied: bool = False  # Track if noise was applied this round


class WSLSBot:
    """
    Win-Stay/Lose-Shift (WSLS) or "Pavlov" strategy bot.

    This bot cooperates if the previous round payoff was satisfactory (≥ mutual cooperation payoff),
    otherwise it shifts to the opposite action. This creates a forgiving strategy that can
    recover from accidental defections and exploit noise in the environment.
    """

    def __init__(self, payoff_matrix: Dict[Tuple[str, str], Tuple[float, float]]):
        self.payoff_matrix = payoff_matrix
        # Use mutual cooperation payoff as the threshold for "winning"
        self.win_threshold = payoff_matrix[("cooperate", "cooperate")][0]
        self.last_action = "cooperate"  # Start cooperatively
        self.last_payoff = None

    def get_action(
        self, my_history: List[str], opponent_history: List[str], round_number: int
    ) -> str:
        """Get WSLS action based on previous round payoff."""
        if round_number == 1 or not my_history:
            # First round: cooperate
            self.last_action = "cooperate"
            return "cooperate"

        # Look up what our payoff was last round
        prev_my_action = my_history[-1]
        prev_opp_action = opponent_history[-1]
        payoff_key = (prev_my_action, prev_opp_action)

        if payoff_key in self.payoff_matrix:
            my_payoff = self.payoff_matrix[payoff_key][0]
        else:
            # Default to defect payoff if invalid
            my_payoff = self.payoff_matrix[("defect", "defect")][0]

        # Win-Stay/Lose-Shift logic
        if my_payoff >= self.win_threshold:
            # Win: stay with the same action
            action = self.last_action
        else:
            # Lose: shift to opposite action
            action = "defect" if self.last_action == "cooperate" else "cooperate"

        self.last_action = action
        self.last_payoff = my_payoff
        return action

    def reset(self):
        """Reset bot state for new game."""
        self.last_action = "cooperate"
        self.last_payoff = None


class IteratedPrisonersDilemma(BaseGame):
    """
    Iterated Prisoner's Dilemma game where players submit strategy code.

    Players compete by writing functions that decide whether to cooperate or defect
    based on the history of previous rounds.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_rounds = config.get("num_rounds", 100)

        # Handle both config formats for payoff matrix
        payoff_config = config.get("payoff_matrix", {})
        if "cooperate_cooperate" in payoff_config:
            # New config format from YAML
            self.payoff_matrix = {
                ("cooperate", "cooperate"): tuple(
                    payoff_config.get("cooperate_cooperate", [3, 3])
                ),
                ("cooperate", "defect"): tuple(
                    payoff_config.get("cooperate_defect", [0, 5])
                ),
                ("defect", "cooperate"): tuple(
                    payoff_config.get("defect_cooperate", [5, 0])
                ),
                ("defect", "defect"): tuple(payoff_config.get("defect_defect", [1, 1])),
            }
        else:
            # Fallback to old tuple format
            self.payoff_matrix = payoff_config or {
                ("cooperate", "cooperate"): (3, 3),
                ("cooperate", "defect"): (0, 5),
                ("defect", "cooperate"): (5, 0),
                ("defect", "defect"): (1, 1),
            }

        # Noise parameters for action flipping
        self.noise_prob = config.get("noise_prob", 0.05)  # 5% default noise
        self.noise_enabled = config.get("noise_enabled", True)

        # WSLS bot configuration
        self.wsls_bot_prob = config.get(
            "wsls_bot_prob", 0.2
        )  # 20% chance of WSLS opponent
        self.wsls_bot_enabled = config.get("wsls_bot_enabled", True)

        # Initialize WSLS bot
        self.wsls_bot = WSLSBot(self.payoff_matrix)

    def get_game_name(self) -> str:
        return "Iterated Prisoner's Dilemma"

    def get_player_prompt(self, player_id: int, role: str) -> str:
        """Generate a prompt for players to write their prisoner's dilemma strategy."""
        noise_info = ""
        if self.noise_enabled:
            noise_info = f"""
IMPORTANT - NOISY ENVIRONMENT: 
There is a {self.noise_prob:.1%} chance each round that any action might be randomly flipped 
(cooperate→defect or defect→cooperate). Design robust strategies that can recover from errors!
"""

        return f"""Write a Python strategy function for an Iterated Prisoner's Dilemma tournament ({self.num_rounds} rounds).

TASK: You must write ONLY a Python function called `strategy`. No explanations, no additional text.

PAYOFFS:
- Both cooperate: 3 points each
- Both defect: 1 point each  
- One cooperates, one defects: 0 for cooperator, 5 for defector{noise_info}
FUNCTION SIGNATURE:
```python
def strategy(my_history, opponent_history, round_number):
    # my_history: list of your actions ['cooperate', 'defect', ...]
    # opponent_history: list of opponent's actions ['cooperate', 'defect', ...]
    # round_number: current round (1, 2, 3, ...)
    # MUST return either 'cooperate' or 'defect'
    
    # Your code here
    return 'cooperate'  # or 'defect'
```

EXAMPLES:
- Tit-for-tat: `if not opponent_history: return 'cooperate'` then `return opponent_history[-1]`
- Always cooperate: `return 'cooperate'`
- Always defect: `return 'defect'`
- Forgiving Tit-for-tat: Cooperate after single defections, defect after consecutive defections

Write your strategy function now (function only, no other text):"""

    def extract_code_from_response(self, response: str) -> str:
        """Extract the strategy function from the LLM response."""
        # Look for code in markdown format
        code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Fallback: look for function definition
        func_match = re.search(
            r"def strategy\(.*?\):(.*?)(?=\n\S|\n*$)", response, re.DOTALL
        )
        if func_match:
            return f"def strategy{func_match.group(0)[len('def strategy'):]}"

        # Last resort: return the response as-is
        return response.strip()

    def validate_submission(
        self, code: str, player_id: int, role: str
    ) -> Tuple[bool, str]:
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

    def _apply_noise(self, action: str) -> Tuple[str, bool]:
        """
        Apply noise to an action with a small probability.

        Returns:
            Tuple of (possibly_flipped_action, was_noise_applied)
        """
        if not self.noise_enabled:
            return action, False

        if np.random.rand() < self.noise_prob:
            # Flip the action
            flipped_action = "defect" if action == "cooperate" else "cooperate"
            return flipped_action, True
        else:
            return action, False

    def _should_use_wsls_bot(self) -> bool:
        """Determine if we should use WSLS bot for one of the players."""
        return self.wsls_bot_enabled and np.random.rand() < self.wsls_bot_prob

    def play_game(self, submissions: List[PlayerSubmission]) -> GameResult:
        """
        Run a complete iterated prisoner's dilemma game.

        The game consists of multiple rounds where each player's strategy function
        is called to get their action, then payoffs are calculated.
        """
        if len(submissions) != 2:
            return GameResult(
                player_rewards={
                    i: self.get_reward_on_execution_failure()
                    for i in range(len(submissions))
                },
                game_data={"error": f"Expected 2 players, got {len(submissions)}"},
                execution_logs=["Invalid number of players"],
                successful_submissions=0,
            )

        # Check if both submissions compiled successfully
        successful_count = sum(1 for sub in submissions if sub.compilation_success)
        if successful_count < 2:
            # If either player has compilation errors, both get failure reward
            rewards = {}
            for sub in submissions:
                if sub.compilation_success:
                    rewards[sub.player_id] = (
                        self.get_reward_on_execution_failure()
                    )  # Penalty for opponent's failure
                else:
                    rewards[sub.player_id] = self.get_reward_on_execution_failure()

            return GameResult(
                player_rewards=rewards,
                game_data={"error": "One or more players had compilation errors"},
                execution_logs=[
                    f"Player {sub.player_id}: {'OK' if sub.compilation_success else sub.compilation_error}"
                    for sub in submissions
                ],
                successful_submissions=successful_count,
            )

        # Initialize game state
        player1, player2 = submissions[0], submissions[1]
        history = {player1.player_id: [], player2.player_id: []}
        total_payoffs = {player1.player_id: 0.0, player2.player_id: 0.0}
        round_results = []
        execution_logs = []

        # Determine if we should use WSLS bot for player 2
        use_wsls_for_p2 = self._should_use_wsls_bot()
        if use_wsls_for_p2:
            self.wsls_bot.reset()
            execution_logs.append("Using WSLS bot for Player 2")

        # Play the iterated game
        for round_num in range(1, self.num_rounds + 1):
            # Get actions from both players
            p1_action = self._get_player_action(
                player1,
                history[player1.player_id],
                history[player2.player_id],
                round_num,
            )

            # Use WSLS bot for player 2 if selected, otherwise use submitted strategy
            if use_wsls_for_p2:
                p2_action = self.wsls_bot.get_action(
                    history[player2.player_id],
                    history[player1.player_id],
                    round_num,
                )
            else:
                p2_action = self._get_player_action(
                    player2,
                    history[player2.player_id],
                    history[player1.player_id],
                    round_num,
                )

            # Handle execution failures
            if p1_action is None or p2_action is None:
                # If either player fails, end the game and give failure rewards
                execution_logs.append(f"Round {round_num}: Strategy execution failed")
                return GameResult(
                    player_rewards={
                        player1.player_id: self.get_reward_on_execution_failure(),
                        player2.player_id: self.get_reward_on_execution_failure(),
                    },
                    game_data={
                        "rounds_completed": round_num - 1,
                        "failure_reason": "Strategy execution failed",
                        "round_results": round_results,
                    },
                    execution_logs=execution_logs,
                    successful_submissions=successful_count,
                )

            # Store original actions for logging
            original_p1_action = p1_action
            original_p2_action = p2_action

            # Apply noise to actions
            p1_action, p1_noise_applied = self._apply_noise(p1_action)
            p2_action, p2_noise_applied = self._apply_noise(p2_action)

            noise_applied_this_round = p1_noise_applied or p2_noise_applied

            # Log noise if applied
            if noise_applied_this_round:
                noise_desc = []
                if p1_noise_applied:
                    noise_desc.append(f"P1: {original_p1_action}→{p1_action}")
                if p2_noise_applied:
                    noise_desc.append(f"P2: {original_p2_action}→{p2_action}")
                execution_logs.append(
                    f"Round {round_num}: Noise applied - {', '.join(noise_desc)}"
                )

            # Calculate payoffs for this round
            payoff_key = (p1_action, p2_action)
            if payoff_key in self.payoff_matrix:
                p1_payoff, p2_payoff = self.payoff_matrix[payoff_key]
            else:
                # Invalid actions default to defect vs defect
                p1_payoff, p2_payoff = self.payoff_matrix[("defect", "defect")]
                execution_logs.append(
                    f"Round {round_num}: Invalid actions, defaulting to defect vs defect"
                )

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
                round_number=round_num,
                noise_applied=noise_applied_this_round,
            )
            round_results.append(round_result)

            execution_logs.append(
                f"Round {round_num}: P1={p1_action}, P2={p2_action}, Payoffs=({p1_payoff}, {p2_payoff})"
            )

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
        execution_logs.append(
            f"Winner: {'P1' if p1_total > p2_total else 'P2' if p2_total > p1_total else 'Tie'}"
        )

        # Count noise applications for statistics
        noise_rounds = sum(1 for result in round_results if result.noise_applied)

        return GameResult(
            player_rewards=rewards,
            game_data={
                "rounds_completed": self.num_rounds,
                "final_payoffs": total_payoffs,
                "round_results": round_results,
                "winner": (
                    "player1"
                    if p1_total > p2_total
                    else "player2" if p2_total > p1_total else "tie"
                ),
                "noise_rounds": noise_rounds,
                "noise_probability": self.noise_prob,
                "wsls_bot_used": use_wsls_for_p2,
            },
            execution_logs=execution_logs,
            successful_submissions=successful_count,
        )

    def _get_player_action(
        self,
        player: PlayerSubmission,
        my_history: List[str],
        opponent_history: List[str],
        round_number: int,
    ) -> str:
        """Get a player's action for the current round."""
        try:
            # Execute the player's strategy function
            result = self.code_executor.execute_strategy_code(
                player.extracted_code,
                "strategy",
                my_history.copy(),  # Pass copies to avoid modification
                opponent_history.copy(),
                round_number,
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
