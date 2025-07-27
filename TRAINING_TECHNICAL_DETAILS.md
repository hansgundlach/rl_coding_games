# Qwen Connect-X PPO Training: Technical Implementation Details

## Overview

This document explains the technical details of how the Qwen2.5-3B model is trained using Proximal Policy Optimization (PPO) for the Connect-X game environment, including rollout collection, parameter updates, evaluation procedures, and system architecture.

## Training Architecture

### Model Setup
- **Base Model**: Qwen/Qwen2.5-3B (3 billion parameter language model)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) with rank 16
- **Target Modules**: All attention and MLP layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- **Device**: Auto-detection (CUDA if available, else CPU)

### LoRA Configuration
```yaml
lora:
  r: 16                    # Low-rank dimension
  lora_alpha: 32          # Scaling factor (α/r = 2.0)
  lora_dropout: 0.1       # Dropout rate
  bias: "none"            # No bias adaptation
```

## Training Loop Structure

### 1. Epoch-Based Training
```
for epoch in range(num_epochs):
    # Phase 1: Rollout Collection
    experiences = collect_rollouts(episodes_per_epoch)
    
    # Phase 2: Model Update
    policy_loss = update_model(experiences)
    
    # Phase 3: Evaluation & Checkpointing
    if epoch % save_interval == 0:
        save_checkpoint()
        evaluate_coding_performance()
```

### 2. Configuration Parameters
From `configs/qwen_ppo.yaml`:
- **num_epochs**: 2 (for testing; typically 50-100)
- **episodes_per_epoch**: 5 (typically 20-50)
- **batch_size**: 2 (typically 8-32)
- **learning_rate**: 5e-5
- **save_interval**: 1 epoch (typically every 5-10 epochs)

## Rollout Collection Process

### Episode Structure
Each episode involves the Qwen model playing Connect-X against a random opponent:

1. **Environment Reset**: 6x7 Connect-X board initialized
2. **Turn-based Gameplay**: 
   - Qwen model plays as player 1 or 2 (alternating)
   - Random opponent provides baseline challenge
3. **Experience Collection**: Each turn generates:
   - `state`: Current board configuration (42-element array)
   - `action`: Column choice (0-6)
   - `reward`: Immediate reward from environment
   - `prompt`: Text representation of board state for Qwen
   - `response`: Model's action as text
   - `log_prob`: Action probability (simplified implementation)

### Prompt Generation
The model receives prompts like:
```
Current Connect-X board (Player 1's turn):
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |
|   |   | X |   |   |   |   |
| O | X | O |   |   |   |   |
Valid moves: [0, 1, 3, 4, 5, 6]
Choose your move (0-6): 
```

### Reward Structure
- **Win**: +100 points
- **Loss**: -100 points  
- **Draw**: 0 points
- **Invalid Move**: -10 points
- **Step Penalty**: -1 per turn (encourages quick wins)
- **Reward Shaping**: Optional heuristic bonuses for strategic positions

## Parameter Update Mechanism

### 1. Reward-to-Go Computation
```python
def compute_rewards_to_go(experiences):
    gamma = 0.99  # Discount factor
    rewards_to_go = []
    running_return = 0
    
    # Backward pass through episode
    for exp in reversed(experiences):
        running_return = exp.reward + gamma * running_return
        rewards_to_go.insert(0, running_return)
    
    return rewards_to_go
```

### 2. Loss Function (Simplified PPO)
The implementation uses a simplified PPO approach:

```python
# Normalize rewards
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

# Language modeling loss weighted by rewards
loss = model(input_ids=tokens, labels=tokens).loss
weighted_loss = loss * (1.0 - rewards.mean())

# Higher rewards → Lower loss → More reinforcement
```

### 3. Mini-batch Updates
- **Batch Processing**: Experiences split into mini-batches of size 2
- **Tokenization**: Prompts + responses concatenated and tokenized (max_length=512)
- **Gradient Clipping**: Max norm of 1.0 to prevent gradient explosion
- **Optimizer**: AdamW with weight decay 0.01

### 4. Update Frequency
- **Parameter Updates**: Once per epoch (after collecting all episodes)
- **Episodes per Update**: 5 episodes → ~50-100 experiences per update
- **Gradient Steps**: Number of mini-batches per epoch

## Evaluation Pipeline

### 1. Game Performance Metrics
Logged every epoch:
- Mean episode reward
- Episode length distribution  
- Win/loss/draw rates
- Policy loss magnitude

### 2. Coding Benchmark Evaluation
Performed every `coding_eval_interval` epochs with two modes:

#### Quick Evaluation Mode (Default)
- **Purpose**: Fast feedback during training
- **HumanEval-Plus**: 10-20 problems (configurable via `quick_eval_size`)
- **MBPP-Plus**: 10-20 problems (configurable via `quick_eval_size`)
- **Evaluation Method**: Simple execution tests (less rigorous than full EvalPlus)
- **Evaluation Time**: ~30-60 seconds
- **Use Case**: Regular monitoring during training

#### Full Evaluation Mode
- **Purpose**: Comprehensive assessment for final results
- **HumanEval-Plus**: All 164 Python programming problems
- **MBPP-Plus**: All 974 Python programming problems  
- **Evaluation Method**: Complete EvalPlus evaluation with extended test suites
- **Temperature**: 0.2 (lower for more deterministic code generation)
- **Max Tokens**: 512
- **Samples**: 1 per problem (pass@1 metric)
- **Evaluation Time**: ~20-30 minutes per run
- **Trigger**: Every `full_eval_interval` epochs (e.g., every 50 epochs)

### 3. Evaluation Configuration
```yaml
evaluation:
  coding_eval_interval: 10    # Run coding eval every 10 epochs
  quick_eval: true           # Use quick mode by default
  quick_eval_size: 20        # 20 problems per dataset for quick eval
  full_eval_interval: 50     # Full evaluation every 50 epochs
  num_coding_samples: 1      # Single sample per problem
```

### 4. Baseline Performance
- **Qwen2.5-3B Base Model**: 1.8% pass@1 on HumanEval-Plus (full evaluation)
- **Quick Evaluation**: ~67% pass rate on 3-problem subset (varies by selection)
- **Expected Improvement**: 5-15% after RL training
- **Evaluation Frequency**: Quick eval every 10 epochs, full eval every 50 epochs

## Memory and Computational Requirements

### Model Memory Usage
- **Base Model**: ~6GB GPU memory (FP16)
- **LoRA Adapters**: ~200MB additional parameters
- **Gradients**: ~400MB during training
- **Total GPU Memory**: ~8-10GB recommended

### Training Time Estimates
- **Single Episode**: 30-60 seconds (including inference)
- **Single Epoch**: 5-10 minutes (5 episodes for testing, 10-20 minutes for production)
- **Quick Coding Evaluation**: 30-60 seconds (10-20 problems)
- **Full Coding Evaluation**: 20-30 minutes (all problems)
- **Testing Training Run**: 30-60 minutes (2 epochs with quick eval)
- **Production Training Run**: 20-30 hours (100 epochs with mixed evaluation)

## Logging and Monitoring

### WandB Integration
All metrics logged to Weights & Biases:

```python
training_stats = {
    "qwen_ppo/loss/policy": policy_loss,
    "episode/mean_reward": mean_reward,
    "episode/mean_length": mean_length,
    "coding/humaneval_plus/pass@1": humaneval_score,
    "coding/mbpp_plus/pass@1": mbpp_score,
    "training/total_steps": total_steps
}
```

### Checkpoint Management
- **Format**: LoRA adapter weights only (not full model)
- **Size**: ~200MB per checkpoint
- **Naming**: `qwen_ppo_epoch_{epoch_number}`
- **Location**: `checkpoints/qwen_ppo/`

## Key Technical Innovations

### 1. Language Model as Policy
Unlike traditional RL which outputs action logits, this approach:
- Generates natural language descriptions of actions
- Uses next-token prediction loss as policy objective
- Leverages pre-trained language understanding

### 2. Dual Objective Training
The model simultaneously optimizes for:
- **Game Performance**: Connect-X win rate and strategic play
- **Code Generation**: Maintaining/improving programming capabilities

### 3. Efficient Fine-tuning
- **LoRA**: Only 0.5% of parameters are trainable
- **Memory Efficient**: Full model stays frozen
- **Transfer Learning**: Preserves general language capabilities

## Potential Issues and Mitigations

### 1. Reward Sparsity
- **Problem**: Few wins against random opponent
- **Solution**: Reward shaping with intermediate bonuses

### 2. Catastrophic Forgetting
- **Problem**: Loss of coding ability during game training
- **Solution**: Regular coding evaluations and mixed training

### 3. Sample Efficiency
- **Problem**: RL requires many episodes to learn
- **Solution**: Warm-start from pre-trained model, small learning rates

### 4. Exploration vs Exploitation
- **Problem**: Model may exploit simple strategies
- **Solution**: Temperature scheduling, diverse opponents

## Configuration Tuning Guidelines

### For Production Training:
```yaml
training:
  num_epochs: 100
  episodes_per_epoch: 20
  batch_size: 8
  learning_rate: 1e-5

evaluation:
  skip_initial_eval: false
  coding_eval_interval: 10
```

### For Debugging:
```yaml
training:
  num_epochs: 5
  episodes_per_epoch: 2  
  batch_size: 1

evaluation:
  skip_initial_eval: true
  coding_eval_interval: 1
```

This architecture represents a novel approach to training language models for strategic reasoning while preserving their general capabilities through continuous evaluation and efficient fine-tuning methods.