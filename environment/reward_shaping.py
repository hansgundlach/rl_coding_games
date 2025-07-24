"""Reward shaping utilities for Connect-X training."""

import numpy as np
from typing import Dict, Any


class RewardShaper:
    """Custom reward shaping for Connect-X RL training."""
    
    def __init__(
        self,
        win_reward: float = 100.0,
        lose_penalty: float = -100.0,
        draw_reward: float = 0.0,
        center_bonus: float = 1.0,
        threat_bonus: float = 5.0,
        block_bonus: float = 3.0,
    ):
        """Initialize reward shaper."""
        self.win_reward = win_reward
        self.lose_penalty = lose_penalty
        self.draw_reward = draw_reward
        self.center_bonus = center_bonus
        self.threat_bonus = threat_bonus
        self.block_bonus = block_bonus
    
    def shape_reward(
        self,
        prev_board: np.ndarray,
        curr_board: np.ndarray,
        action: int,
        base_reward: float,
        done: bool,
        info: Dict[str, Any] = None,
    ) -> float:
        """Apply reward shaping to base reward."""
        shaped_reward = base_reward
        
        # Terminal rewards
        if done:
            if base_reward > 0:  # Win
                shaped_reward = self.win_reward
            elif base_reward < 0:  # Lose
                shaped_reward = self.lose_penalty
            else:  # Draw
                shaped_reward = self.draw_reward
            return shaped_reward
        
        # Center column bonus
        rows, cols = curr_board.shape
        center_col = cols // 2
        if abs(action - center_col) <= 1:
            shaped_reward += self.center_bonus
        
        # Check for creating threats or blocking opponent
        player_mark = 1  # Assume we are player 1
        opponent_mark = 2
        
        # Bonus for creating winning threats
        if self._creates_threat(curr_board, action, player_mark):
            shaped_reward += self.threat_bonus
            
        # Bonus for blocking opponent threats
        if self._blocks_threat(prev_board, curr_board, action, opponent_mark):
            shaped_reward += self.block_bonus
        
        return shaped_reward
    
    def _creates_threat(self, board: np.ndarray, action: int, player_mark: int) -> bool:
        """Check if the action creates a winning threat."""
        rows, cols = board.shape
        
        # Find the row where the piece was placed
        row = -1
        for r in range(rows):
            if board[r, action] == player_mark:
                row = r
                break
        
        if row == -1:
            return False
        
        # Check if this creates a 3-in-a-row that can become 4
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonals
        
        for dr, dc in directions:
            count = 1  # Count the piece we just placed
            
            # Check positive direction
            r, c = row + dr, action + dc
            while 0 <= r < rows and 0 <= c < cols and board[r, c] == player_mark:
                count += 1
                r += dr
                c += dc
            
            # Check negative direction
            r, c = row - dr, action - dc
            while 0 <= r < rows and 0 <= c < cols and board[r, c] == player_mark:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 3:
                return True
        
        return False
    
    def _blocks_threat(self, prev_board: np.ndarray, curr_board: np.ndarray, 
                      action: int, opponent_mark: int) -> bool:
        """Check if the action blocks an opponent threat."""
        # Simple heuristic: if opponent had a 3-in-a-row that we interrupted
        rows, cols = prev_board.shape
        
        # Check if opponent had threatening positions that we blocked
        for r in range(rows):
            for c in range(cols):
                if prev_board[r, c] == opponent_mark:
                    # Check if placing our piece at 'action' column blocks a potential win
                    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                    for dr, dc in directions:
                        count = 0
                        # Count consecutive opponent pieces
                        nr, nc = r, c
                        while 0 <= nr < rows and 0 <= nc < cols and prev_board[nr, nc] == opponent_mark:
                            count += 1
                            nr += dr
                            nc += dc
                        
                        if count >= 2 and nc == action:  # Simplified blocking check
                            return True
        
        return False