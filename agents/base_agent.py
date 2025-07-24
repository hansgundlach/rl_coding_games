"""Base agent interface for Connect-X."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for Connect-X agents."""
    
    def __init__(self, name: str = "BaseAgent"):
        """Initialize the agent.
        
        Args:
            name: Agent identifier name.
        """
        self.name = name
        self.player_id: Optional[int] = None
    
    @abstractmethod
    def act(self, observation: Dict[str, Any], configuration: Dict[str, Any]) -> int:
        """Select an action given the current observation.
        
        Args:
            observation: Game state observation from kaggle_environments.
            configuration: Game configuration parameters.
            
        Returns:
            Column index (0-based) for the next move.
        """
        pass
    
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        pass
    
    def get_valid_actions(self, board: np.ndarray) -> List[int]:
        """Get list of valid column indices for the current board state.
        
        Args:
            board: 2D numpy array representing the game board.
            
        Returns:
            List of valid column indices.
        """
        return [col for col in range(board.shape[1]) if board[0, col] == 0]