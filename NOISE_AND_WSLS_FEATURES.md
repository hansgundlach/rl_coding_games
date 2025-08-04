# SPIRAL Prisoner's Dilemma: Noise and WSLS Bot Features

## Overview

Enhanced the SPIRAL Prisoner's Dilemma implementation with two key features to improve training dynamics and strategy diversity:

### 1. Action Noise (5% default)

- **What**: Small probability that any player action gets randomly flipped (cooperate ↔ defect)
- **Why**: Breaks stability of mutual defection, makes cooperation profitable, prevents gradient signals from vanishing
- **Implementation**: Applied after strategy execution but before payoff calculation
- **Research basis**: Classic studies show 2-5% noise enables forgiving strategies like WSLS and Generous Tit-for-Tat to dominate

### 2. WSLS (Win-Stay/Lose-Shift) Bot

- **What**: Pavlov-style bot that cooperates if previous payoff ≥ mutual cooperation threshold, otherwise switches action
- **Why**: Provides forgiving baseline that exploits noise, teaches LLMs cooperative patterns
- **Implementation**: Replaces Player 2 with 20% probability (configurable)
- **Strategy**: Start with cooperation, then Win-Stay/Lose-Shift based on previous round payoff

## Configuration

### In `configs/spiral_prisoners_dilemma.yaml`

```yaml
game:
  # Noise settings
  noise_enabled: true           # Enable action noise
  noise_prob: 0.05             # 5% probability of flipping each action (0.03-0.08 recommended)
  
  # WSLS bot settings
  wsls_bot_enabled: true       # Enable WSLS bot as occasional opponent
  wsls_bot_prob: 0.2           # 20% chance of facing WSLS bot instead of another LLM
```

## Key Benefits

1. **Richer Payoff Landscape**: Noise creates opportunities for forgiveness and cooperation
2. **Gradient Signal Preservation**: Prevents vanishing gradients from stable mutual defection
3. **Strategy Diversity**: Encourages development of robust, error-correcting strategies
4. **Cooperative Seed**: WSLS bot provides clear cooperative exemplar to learn from

## Statistics Tracked

- `noise_rounds`: Number of rounds where noise was applied
- `noise_game_rate`: Fraction of games affected by noise
- `wsls_bot_rate`: Fraction of games using WSLS bot
- `wsls_bot_win_rate`: Win rate of WSLS bot when used

## Player Awareness

Players are informed about the noisy environment in their prompt:
> "IMPORTANT - NOISY ENVIRONMENT: There is a 5.0% chance each round that any action might be randomly flipped (cooperate→defect or defect→cooperate). Design robust strategies that can recover from errors!"

This encourages development of forgiving strategies that can handle occasional execution errors.

## Research References

- Nature studies on noise in evolutionary game theory
- JSTOR publications on Generous Tit-for-Tat in noisy environments  
- Computer Science at UMD research on WSLS strategy performance
- Forbes analysis of tournament dynamics with noise
