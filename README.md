# Q-Learning Snake Game

A Snake game implementation that learns to play itself using Q-learning reinforcement learning algorithm.

## Requirements

```bash
pip install pygame
```

## How to Run

```bash
python snake.py
```

Choose option 1 to train a new agent, or option 2 to watch a trained agent play.

## Pygame Basics

Pygame is used only for visualization in demo mode. The training happens without graphics for speed.

**Key components:**
- `pygame.display.set_mode()` creates the game window
- `screen.fill()` and `pygame.draw.rect()` render the snake and food
- `clock.tick(FPS)` controls game speed
- Event loop handles user input (quit, restart)

The game logic is separate from visualization, so training runs fast without rendering each frame.

## Q-Learning Explanation

Q-learning is a reinforcement learning algorithm where an agent learns by trial and error. The agent doesn't know the rules initially but learns from rewards and penalties.

### State Representation

Instead of using the entire 10x10 grid, we use 8 simple features:
- 3 danger indicators: Is there danger straight ahead, to the left, or to the right?
- 4 food direction indicators: Is food up, down, left, or right from the snake's head?
- 1 current direction: Which way is the snake facing (up, right, down, left)?

This small state space makes learning much faster than using raw grid positions.

### Actions

The snake has 3 actions:
- Go straight
- Turn left
- Turn right

These are relative to current direction, not absolute. This makes the policy rotation-invariant.

### Rewards

- +10 for eating food
- -10 for dying (hitting wall or self)
- -0.01 for each step (encourages efficiency)

### The Q-Table

The Q-table is a lookup table that stores quality values for each state-action pair. Think of it as a cheat sheet that says "when in this situation, how good is this action?"

Structure: Q[state][action] = expected future reward

Initially all values are 0. As the agent explores, these values get updated based on experience.

### Learning Process

**Epsilon-Greedy Exploration:**
- Start with epsilon = 1.0 (100% random actions)
- Gradually decay to epsilon = 0.01 (1% random)
- Random actions help discover new strategies
- Greedy actions exploit what's already learned

**Q-Value Update Rule:**

At each step, we update the Q-value using:

```
new_Q(s,a) = old_Q(s,a) + alpha * [reward + gamma * max_Q(s',a') - old_Q(s,a)]
```

Where:
- `s` = current state
- `a` = action taken
- `s'` = next state after action
- `alpha` = learning rate (0.1) - how much to adjust based on new info
- `gamma` = discount factor (0.9) - how much to value future rewards
- `reward` = immediate reward received

**What this means:**
The algorithm adjusts its estimate of an action's value based on:
1. The immediate reward received
2. The best possible value from the next state
3. How different this is from the current estimate

Over thousands of episodes, the Q-table converges to optimal values.

### Training Process

1. Start episode with snake in center
2. Observe current state
3. Choose action (random or best from Q-table)
4. Execute action and observe reward and next state
5. Update Q-table using the formula above
6. Repeat until snake dies
7. Start new episode

After many episodes, the agent learns patterns like:
- Move toward food when safe
- Avoid walls and self
- Take efficient paths

### Stopping Criteria

Training stops when:
- Maximum episodes reached (10,000), OR
- Performance plateaus (average score doesn't improve for 2,500 episodes)

### Demo Mode

After training:
- Load saved Q-table
- Set epsilon = 0 (no exploration, pure exploitation)
- Snake always picks the best action according to learned Q-values
- Visualize the learned behavior

## Results

After training, the agent typically learns to:
- Consistently find and eat food
- Avoid obvious dangers
- Achieve scores of 10-20+ foods per game

The simple state representation and small action space make this problem tractable for tabular Q-learning.