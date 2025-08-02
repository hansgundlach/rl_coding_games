# Game Environments Framework

An extensible framework for SPIRAL self-play training where LLMs compete by submitting code strategies for various games.

## 🎮 Available Games

### 1. Iterated Prisoner's Dilemma (`prisoners_dilemma.py`)
- **Description**: Classic prisoner's dilemma over multiple rounds
- **Strategy**: Players write functions that decide to cooperate or defect based on game history
- **Rewards**: Winner of tournament gets +1, loser gets -1, tie gets 0
- **Failure Penalty**: -1 for code execution failures

### 2. Example: Sorting Challenge (`example_new_game.py`)
- **Description**: Demonstrates framework extensibility with a sorting algorithm competition
- **Strategy**: Players write sorting algorithms competing on correctness and speed
- **Purpose**: Template for implementing new games

## 🏗️ Framework Architecture

### Core Components

1. **`BaseGame`** (`base_game.py`): Abstract base class for all games
2. **`CodeExecutor`**: Safe code execution environment with timeout protection
3. **`PlayerSubmission`**: Data structure for player code submissions
4. **`GameResult`**: Standardized game outcome format

### Key Features

- **Safe Code Execution**: Sandboxed environment with timeout protection
- **Automatic Code Extraction**: Extracts code from LLM responses
- **Validation System**: Checks code compilation before execution
- **Extensible Design**: Easy to add new games following the same pattern

## 🚀 Usage

### Running Prisoner's Dilemma Training

```bash
# Local execution
python spiral_prisoners_dilemma.py --config configs/spiral_prisoners_dilemma.yaml

# SLURM submission  
sbatch submit_spiral_prisoners_dilemma.sh

# With config overrides
sbatch submit_spiral_prisoners_dilemma.sh --training.num_steps=50 --game.num_rounds=50
```

### Configuration Options

#### Game Settings (`configs/spiral_prisoners_dilemma.yaml`)
```yaml
game:
  timeout: 5              # Code execution timeout per strategy call
  num_rounds: 100         # Rounds per prisoner's dilemma game
  payoff_matrix:          # Reward structure
    cooperate_cooperate: [3, 3]
    cooperate_defect: [0, 5]
    defect_cooperate: [5, 0]
    defect_defect: [1, 1]
```

#### Training Parameters
```yaml
training:
  num_steps: 100                    # Training steps
  games_per_step_v100: 2           # Games per step (V100)
  games_per_step_other: 4          # Games per step (A100+)
  learning_rate: 0.00002           # Learning rate
  rae_alpha: 0.95                  # Role-conditioned advantage estimation
```

## 🛠️ Adding New Games

### Step 1: Implement Game Class

Create a new file `game_environments/your_game.py`:

```python
from .base_game import BaseGame, PlayerSubmission, GameResult

class YourGame(BaseGame):
    def get_game_name(self) -> str:
        return "Your Game Name"
        
    def get_player_prompt(self, player_id: int, role: str) -> str:
        # Return prompt for LLM to write strategy code
        pass
        
    def extract_code_from_response(self, response: str) -> str:
        # Extract code from LLM response
        pass
        
    def validate_submission(self, code: str, player_id: int, role: str) -> Tuple[bool, str]:
        # Validate the submitted code
        pass
        
    def play_game(self, submissions: List[PlayerSubmission]) -> GameResult:
        # Run the actual game and return results
        pass
```

### Step 2: Create Training Script

Copy and modify `spiral_prisoners_dilemma.py`:

```python
# Import your game
from game_environments.your_game import YourGame

# Initialize game environment
game_env = YourGame(config["game"])

# The rest of the training loop remains the same
```

### Step 3: Configuration File

Create `configs/spiral_your_game.yaml` with game-specific settings.

### Step 4: SLURM Script

Copy and modify `submit_spiral_prisoners_dilemma.sh` with appropriate names.

## 📊 Game Requirements

All games must implement:

1. **Strategy Function**: Players write a specific function (e.g., `strategy`, `sort_algorithm`)
2. **Deterministic Rewards**: Clear win/loss/tie conditions  
3. **Execution Safety**: Handle code failures gracefully
4. **Timeout Protection**: Prevent infinite loops
5. **Role Support**: Support for different player roles if needed

## 🔒 Safety Features

- **Sandboxed Execution**: Code runs in isolated subprocess
- **Timeout Protection**: Configurable timeout per code execution
- **Memory Limits**: Minimal Python environment
- **Error Handling**: Graceful handling of compilation/runtime errors
- **Input Sanitization**: Safe handling of LLM-generated code

## 📈 Monitoring & Logging

- **W&B Integration**: Automatic logging of training metrics
- **MBPP Evaluation**: Periodic evaluation of general coding ability
- **Game-Specific Metrics**: Win rates, execution success rates, strategy diversity
- **Detailed Logs**: Code submissions, execution results, game outcomes

## 🎯 Strategy Examples for Prisoner's Dilemma

### Tit-for-Tat
```python
def strategy(my_history, opponent_history, round_number):
    if round_number == 1:
        return 'cooperate'  # Start nice
    else:
        return opponent_history[-1]  # Copy opponent's last move
```

### Generous Tit-for-Tat  
```python
def strategy(my_history, opponent_history, round_number):
    import random
    if round_number == 1:
        return 'cooperate'
    elif opponent_history[-1] == 'defect':
        # 90% chance to retaliate, 10% chance to forgive
        return 'defect' if random.random() < 0.9 else 'cooperate'
    else:
        return 'cooperate'
```

### Pavlov (Win-Stay, Lose-Shift)
```python
def strategy(my_history, opponent_history, round_number):
    if round_number == 1:
        return 'cooperate'
    
    # Check last round outcome
    my_last = my_history[-1]
    opp_last = opponent_history[-1]
    
    # Win-stay: if got good payoff (3 or 5), repeat
    # Lose-shift: if got bad payoff (0 or 1), switch
    if (my_last == 'cooperate' and opp_last == 'cooperate') or \
       (my_last == 'defect' and opp_last == 'cooperate'):
        return my_last  # Stay
    else:
        return 'defect' if my_last == 'cooperate' else 'cooperate'  # Shift
```

This framework enables rich exploration of strategic behavior through code-based competition!