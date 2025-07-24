"""Proximal Policy Optimization (PPO) algorithm implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class PPOExperience:
    """Single PPO experience."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class PPOBuffer:
    """Experience buffer for PPO."""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.advantages = []
        self.returns = []
    
    def add(self, experience: PPOExperience):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def compute_gae(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation."""
        if len(self.buffer) == 0:
            return
            
        advantages = []
        returns = []
        
        # Convert to lists for easier processing
        experiences = list(self.buffer)
        
        # Compute advantages using GAE
        gae = 0
        for i in reversed(range(len(experiences))):
            exp = experiences[i]
            
            if i == len(experiences) - 1:
                next_value = 0 if exp.done else exp.value
            else:
                next_value = experiences[i + 1].value
            
            delta = exp.reward + gamma * next_value - exp.value
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + exp.value)
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages.tolist()
        self.returns = returns
    
    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Get random batch from buffer."""
        if len(self.buffer) < batch_size:
            indices = list(range(len(self.buffer)))
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        actions = []
        old_log_probs = []
        returns = []
        advantages = []
        
        for i in indices:
            exp = self.buffer[i]
            states.append(exp.state.flatten())
            actions.append(exp.action)
            old_log_probs.append(exp.log_prob)
            returns.append(self.returns[i] if i < len(self.returns) else 0)
            advantages.append(self.advantages[i] if i < len(self.advantages) else 0)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(old_log_probs),
            torch.FloatTensor(returns),
            torch.FloatTensor(advantages)
        )
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.advantages = []
        self.returns = []
    
    def __len__(self):
        return len(self.buffer)


class PolicyNetwork(nn.Module):
    """Policy network for PPO."""
    
    def __init__(self, input_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[int, float]:
        """Get action and its log probability."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()


class ValueNetwork(nn.Module):
    """Value network for PPO."""
    
    def __init__(self, input_size: int, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


class PPOAgent:
    """PPO Agent."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.policy_net = PolicyNetwork(state_size, action_size).to(device)
        self.value_net = ValueNetwork(state_size).to(device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = PPOBuffer()
    
    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Get action, log probability, and value estimate."""
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action_and_log_prob(state_tensor)
            value = self.value_net(state_tensor).item()
        
        return action, log_prob, value
    
    def store_experience(self, experience: PPOExperience):
        """Store experience in buffer."""
        self.buffer.add(experience)
    
    def update(self, ppo_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        """Update policy and value networks."""
        if len(self.buffer) == 0:
            return {}
        
        # Compute GAE
        self.buffer.compute_gae(self.gamma, self.gae_lambda)
        
        stats = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'kl_div': []}
        
        for _ in range(ppo_epochs):
            # Get batch
            states, actions, old_log_probs, returns, advantages = self.buffer.get_batch(batch_size)
            states = states.to(self.device)
            actions = actions.to(self.device)
            old_log_probs = old_log_probs.to(self.device)
            returns = returns.to(self.device)
            advantages = advantages.to(self.device)
            
            # Policy update
            logits = self.policy_net(states)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Compute policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Value update
            values = self.value_net(states)
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            # Compute KL divergence for monitoring
            with torch.no_grad():
                kl_div = (old_log_probs - new_log_probs).mean()
            
            # Store stats
            stats['policy_loss'].append(policy_loss.item())
            stats['value_loss'].append(value_loss.item())
            stats['entropy'].append(entropy.item())
            stats['kl_div'].append(kl_div.item())
        
        # Clear buffer
        self.buffer.clear()
        
        # Return averaged stats
        return {k: np.mean(v) for k, v in stats.items()}
    
    def save(self, filepath: str):
        """Save model parameters."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])