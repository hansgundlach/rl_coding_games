# Project on RL Training of Language Models with Adversarial Games


What happens when models strategize about the iterated prisoners dilemma?
Here we design a scenario where models generate python code strategies to play the iterated prisoners dilemma game. We add a variety of techniques to increase the complexity of the iterated prisoners dilemma game. Here is a sample setting file. We add a Win-Stay, Lose-Shift Bot (WSLS) bot with a 20% frequency to stabilize and make environment more stationary. Here is a config sample. We run GRPO as our main training algorithmn. 

```
# Game-specific settings
game:
  timeout: 3  # Code execution timeout in seconds per strategy call
  num_rounds: 20  # Number of rounds in each prisoner's dilemma game
  payoff_matrix:
    # Payoff matrix: (my_action, opponent_action): (my_payoff, opponent_payoff)
    cooperate_cooperate: [3, 3]
    cooperate_defect: [0, 5]
    defect_cooperate: [5, 0]
    defect_defect: [1, 1]
  
  # Noise settings - small chance to flip actions
  noise_enabled: true           # Enable action noise
  noise_prob: 0.05             # 5% probability of flipping each action (0.03-0.08 recommended)
  
  # WSLS (Win-Stay/Lose-Shift) bot settings
  wsls_bot_enabled: true       # Enable WSLS bot as occasional opponent
  wsls_bot_prob: 0.2           # 20% chance of facing WSLS bot instead of frozen copy
```

## Installation: 
Setup venv on machine using requirements.txt
run training by typing:

sbatch submit_grpo_prisoners_dilemma.sh

Built for MIT Supercloud V100s

