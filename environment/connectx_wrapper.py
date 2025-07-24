"""Kaggle Connect-X environment wrapper for RL training."""

import numpy as np
import gym
from gym import spaces
from kaggle_environments import make
from typing import Dict, Any, Tuple, Optional


class ConnectXWrapper(gym.Env):
    """Gym wrapper for Kaggle Connect-X environment."""
    
    def __init__(
        self,
        opponent_agent: str = "random",
        reward_shaper: Optional[Any] = None,
        rows: int = 6,
        columns: int = 7,
        inarow: int = 4,
    ):
        """Initialize Connect-X wrapper."""
        super().__init__()
        
        self.rows = rows
        self.columns = columns
        self.inarow = inarow
        self.opponent_agent = opponent_agent
        self.reward_shaper = reward_shaper
        
        # Create Kaggle environment
        self.env = make(
            "connectx",
            configuration={"rows": rows, "columns": columns, "inarow": inarow},
            debug=False,
        )
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(columns)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(rows, columns), dtype=np.int32
        )
        
        self.trainer = self.env.train([None, opponent_agent])
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.obs = self.trainer.reset()
        return self._get_board_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Validate action
        board = self._get_board_state()
        if not self._is_valid_action(action, board):
            # Invalid move penalty
            reward = -10.0
            return board, reward, True, {"invalid_move": True}
        
        # Take action
        self.obs, reward, done, info = self.trainer.step(action)
        new_board = self._get_board_state()
        
        # Apply reward shaping if available
        if self.reward_shaper is not None:
            reward = self.reward_shaper.shape_reward(board, new_board, action, reward, done)
        
        return new_board, float(reward), done, info
    
    def _get_board_state(self) -> np.ndarray:
        """Extract board state from observation."""
        if self.obs is None:
            return np.zeros((self.rows, self.columns), dtype=np.int32)
        
        board = np.array(self.obs['board']).reshape(self.rows, self.columns)
        return board.astype(np.int32)
    
    def _is_valid_action(self, action: int, board: np.ndarray) -> bool:
        """Check if action is valid."""
        if not (0 <= action < self.columns):
            return False
        return board[0, action] == 0