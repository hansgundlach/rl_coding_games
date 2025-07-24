"""PPO training pipeline for Connect-X agent."""

import os
import yaml
import argparse
import torch
import numpy as np
from datetime import datetime
from training.wandb_logger import WandBLogger
from training.ppo_algorithm import PPOAgent, PPOExperience
from environment.connectx_wrapper import ConnectXWrapper
from environment.reward_shaping import RewardShaper


class ConnectXPPOTrainer:
    """PPO trainer for Connect-X."""
    
    def __init__(self, config_path: str):
        """Initialize PPO trainer."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"Loaded config from {config_path}")
        self.setup_logging()
        self.setup_environment()
        self.setup_agent()
    
    def setup_logging(self):
        """Initialize Weights & Biases logging."""
        self.logger = WandBLogger(
            project=self.config['logging']['project'],
            entity=self.config['logging'].get('entity'),
            name=f"connectx-ppo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=self.config,
        )
    
    def setup_environment(self):
        """Setup Connect-X environment."""
        env_config = self.config['environment']
        reward_shaper = RewardShaper() if env_config.get('reward_shaping', False) else None
        
        self.env = ConnectXWrapper(
            opponent_agent="random",
            reward_shaper=reward_shaper,
            rows=env_config['rows'],
            columns=env_config['columns'],
        )
        print("Environment setup completed")
    
    def setup_agent(self):
        """Setup PPO agent."""
        training_config = self.config['training']
        ppo_config = self.config['ppo']
        
        # Calculate state and action sizes
        state_size = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        action_size = self.env.action_space.n
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        self.agent = PPOAgent(
            state_size=state_size,
            action_size=action_size,
            lr=training_config['learning_rate'],
            gamma=ppo_config['gamma'],
            gae_lambda=ppo_config['gae_lambda'],
            clip_epsilon=ppo_config['clip_epsilon'],
            value_loss_coef=ppo_config['value_loss_coef'],
            entropy_coef=ppo_config['entropy_coef'],
            max_grad_norm=training_config['max_grad_norm'],
            device=device
        )
        print("PPO agent setup completed")
    
    def collect_episode(self) -> tuple:
        """Collect one episode of experience."""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        experiences = []
        
        while True:
            # Get action from agent
            action, log_prob, value = self.agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            experience = PPOExperience(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=done,
                log_prob=log_prob,
                value=value
            )
            experiences.append(experience)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            if done or episode_length >= 100:  # Max episode length
                break
        
        return experiences, episode_reward, episode_length
    
    def train(self):
        """Run PPO training loop."""
        print("Starting actual PPO training...")
        training_config = self.config['training']
        ppo_config = self.config['ppo']
        
        total_steps = 0
        
        for epoch in range(training_config['num_epochs']):
            print(f"Epoch {epoch + 1}/{training_config['num_epochs']}")
            
            # Collect episodes
            epoch_rewards = []
            epoch_lengths = []
            
            for episode in range(training_config['episodes_per_epoch']):
                experiences, episode_reward, episode_length = self.collect_episode()
                
                # Store experiences in agent buffer
                for exp in experiences:
                    self.agent.store_experience(exp)
                
                epoch_rewards.append(episode_reward)
                epoch_lengths.append(episode_length)
                total_steps += episode_length
                
                if (episode + 1) % 5 == 0:
                    print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}")
            
            # Update agent
            if len(self.agent.buffer) > 0:
                update_stats = self.agent.update(
                    ppo_epochs=ppo_config['ppo_epochs'],
                    batch_size=training_config['batch_size']
                )
                
                # Log training statistics
                stats = {
                    "ppo/loss/policy": update_stats.get('policy_loss', 0),
                    "ppo/loss/value": update_stats.get('value_loss', 0),
                    "ppo/entropy": update_stats.get('entropy', 0),
                    "ppo/mean_kl": update_stats.get('kl_div', 0),
                    "episode/mean_reward": np.mean(epoch_rewards),
                    "episode/mean_length": np.mean(epoch_lengths),
                    "episode/min_reward": np.min(epoch_rewards),
                    "episode/max_reward": np.max(epoch_rewards),
                    "training/total_steps": total_steps,
                }
                
                self.logger.log_training_stats(stats, epoch)
                
                print(f"  Mean reward: {np.mean(epoch_rewards):.2f} ± {np.std(epoch_rewards):.2f}")
                print(f"  Policy loss: {update_stats.get('policy_loss', 0):.4f}")
                print(f"  Value loss: {update_stats.get('value_loss', 0):.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoints'].get('save_interval', 5) == 0:
                checkpoint_dir = self.config['checkpoints']['save_dir']
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"ppo_epoch_{epoch + 1}.pt")
                self.agent.save(checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
        
        print("Training completed!")
        self.logger.finish()


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Connect-X PPO agent")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/ppo.yaml",
        help="Path to training configuration file"
    )
    
    args = parser.parse_args()
    
    trainer = ConnectXPPOTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()