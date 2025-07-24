"""Random baseline agent for Connect-X."""

import numpy as np
from typing import Any, Dict
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Random baseline agent for Connect-X."""
    
    def __init__(self, seed: int = 42):
        """Initialize random agent.
        
        Args:
            seed: Random seed for reproducibility.
        """
        super().__init__(name="RandomAgent")
        self.rng = np.random.RandomState(seed)
    
    def act(self, observation: Dict[str, Any], configuration: Dict[str, Any]) -> int:
        """Select random valid action.
        
        Args:
            observation: Game state observation.
            configuration: Game configuration.
            
        Returns:
            Random valid column index.
        """
        board = np.array(observation['board']).reshape(
            configuration['rows'], configuration['columns']
        )
        valid_actions = self.get_valid_actions(board)
        
        if not valid_actions:
            return 0  # Fallback
        
        return self.rng.choice(valid_actions)
    
    def reset(self) -> None:
        """Reset random state."""
        pass