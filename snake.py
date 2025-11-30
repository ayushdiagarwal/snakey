import pygame
import random
import pickle
import os
from collections import deque

# Constants
GRID_SIZE = 10
CELL_SIZE = 30
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 10

# Colors
BLACK = (26, 26, 26)
GRID_COLOR = (42, 42, 42)
GREEN = (34, 197, 94)
DARK_GREEN = (22, 163, 74)
RED = (239, 68, 68)
WHITE = (255, 255, 255)

# RL Hyperparameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
MAX_EPISODES = 10000
CHECK_INTERVAL = 500
PLATEAU_CHECKS = 5

class SnakeGame:
    def __init__(self):
        self.q_table = {}
        self.reset()
    
    def reset(self):
        center = GRID_SIZE // 2
        self.snake = [[center, center], [center, center - 1], [center, center - 2]]
        self.direction = 0  # 0:up, 1:right, 2:down, 3:left
        self.food = self.spawn_food()
        self.score = 0
        self.steps = 0
        self.max_steps = GRID_SIZE * GRID_SIZE * 2
        return self.get_state()
    
    def spawn_food(self):
        while True:
            food = [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]
            if food not in self.snake:
                return food
    
    def is_danger(self, direction):
        head = self.snake[0]
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dy, dx = moves[direction]
        new_y, new_x = head[0] + dy, head[1] + dx
        
        if new_y < 0 or new_y >= GRID_SIZE or new_x < 0 or new_x >= GRID_SIZE:
            return 1
        if [new_y, new_x] in self.snake[:-1]:
            return 1
        return 0
    
    def get_state(self):
        head = self.snake[0]
        dir_map = [
            self.direction,
            (self.direction + 1) % 4,
            (self.direction + 3) % 4
        ]
        
        state = (
            self.is_danger(dir_map[0]),  # danger straight
            self.is_danger(dir_map[1]),  # danger right
            self.is_danger(dir_map[2]),  # danger left
            1 if head[0] > self.food[0] else 0,  # food up
            1 if head[0] < self.food[0] else 0,  # food down
            1 if head[1] > self.food[1] else 0,  # food left
            1 if head[1] < self.food[1] else 0,  # food right
            self.direction
        )
        return state
    
    def step(self, action):
        # action: 0=straight, 1=right, 2=left
        if action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction + 3) % 4
        
        head = self.snake[0]
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dy, dx = moves[self.direction]
        new_head = [head[0] + dy, head[1] + dx]
        
        # Check death
        if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or 
            new_head[1] < 0 or new_head[1] >= GRID_SIZE or 
            new_head in self.snake):
            return self.get_state(), -10, True, self.score
        
        self.snake.insert(0, new_head)
        self.steps += 1
        
        # Check food
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
            self.steps = 0
            return self.get_state(), 10, False, self.score
        else:
            self.snake.pop()
        
        # Check timeout
        if self.steps >= self.max_steps:
            return self.get_state(), -10, True, self.score
        
        return self.get_state(), -0.01, False, self.score
    
    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        return self.q_table[state][action]
    
    def set_q_value(self, state, action, value):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        self.q_table[state][action] = value
    
    def get_best_action(self, state):
        q_values = [self.get_q_value(state, a) for a in range(3)]
        return q_values.index(max(q_values))
    
    def render(self, screen):
        screen.fill(BLACK)
        
        # Draw grid
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(screen, GRID_COLOR, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE))
            pygame.draw.line(screen, GRID_COLOR, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE))
        
        # Draw food
        pygame.draw.rect(screen, RED, (
            self.food[1] * CELL_SIZE + 2,
            self.food[0] * CELL_SIZE + 2,
            CELL_SIZE - 4,
            CELL_SIZE - 4
        ))
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            color = GREEN if i == 0 else DARK_GREEN
            pygame.draw.rect(screen, color, (
                segment[1] * CELL_SIZE + 2,
                segment[0] * CELL_SIZE + 2,
                CELL_SIZE - 4,
                CELL_SIZE - 4
            ))

def train_agent(demo_interval=None):
    print("Starting training...")
    game = SnakeGame()
    epsilon = EPSILON_START
    scores = []
    best_avg = 0
    plateau_count = 0
    
    for episode in range(MAX_EPISODES):
        state = game.reset()
        done = False
        episode_score = 0
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                action = game.get_best_action(state)
            
            next_state, reward, done, score = game.step(action)
            episode_score = score
            
            # Q-learning update
            old_q = game.get_q_value(state, action)
            max_next_q = max([game.get_q_value(next_state, a) for a in range(3)])
            new_q = old_q + ALPHA * (reward + GAMMA * max_next_q - old_q)
            game.set_q_value(state, action, new_q)
            
            state = next_state
        
        scores.append(episode_score)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            recent_scores = scores[-100:]
            avg_score = sum(recent_scores) / len(recent_scores)
            best_score = max(scores)
            print(f"Episode {episode + 1}/{MAX_EPISODES} | "
                  f"Score: {episode_score} | "
                  f"Avg(100): {avg_score:.2f} | "
                  f"Best: {best_score} | "
                  f"ε: {epsilon:.3f}")
        
        # Demo at interval
        if demo_interval and (episode + 1) % demo_interval == 0:
            print(f"\n--- Demo after {episode + 1} episodes ---")
            demo_choice = input("Watch demo now? (y/n): ").strip().lower()
            if demo_choice == 'y':
                # Save current Q-table temporarily
                with open('q_table.pkl', 'wb') as f:
                    pickle.dump(game.q_table, f)
                demo_agent()
                print("\nResuming training...\n")
        
        # Check for plateau
        if (episode + 1) % CHECK_INTERVAL == 0 and episode > 0:
            recent_avg = sum(scores[-100:]) / 100
            if recent_avg - best_avg < 0.1:
                plateau_count += 1
                if plateau_count >= PLATEAU_CHECKS:
                    print(f"\nEarly stopping at episode {episode + 1} (plateau detected)")
                    break
            else:
                best_avg = recent_avg
                plateau_count = 0
    
    # Save Q-table
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(game.q_table, f)
    
    print(f"\nTraining complete!")
    print(f"Final avg score: {sum(scores[-100:]) / 100:.2f}")
    print(f"Best score: {max(scores)}")
    print(f"Q-table saved to q_table.pkl")
    
    return game

def demo_agent():
    if not os.path.exists('q_table.pkl'):
        print("No trained Q-table found. Please train first.")
        return
    
    print("Loading trained agent...")
    game = SnakeGame()
    
    with open('q_table.pkl', 'rb') as f:
        game.q_table = pickle.load(f)
    
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption('Q-Learning Snake - Demo')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    print("Starting demo (press Q to quit, R to restart)...")
    
    running = True
    state = game.reset()
    done = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    state = game.reset()
                    done = False
        
        if not done:
            action = game.get_best_action(state)
            state, reward, done, score = game.step(action)
            
            if done:
                print(f"Game Over! Final Score: {game.score}")
        
        game.render(screen)
        
        # Draw score
        score_text = font.render(f'Score: {game.score}', True, WHITE)
        screen.blit(score_text, (10, 10))
        
        if done:
            game_over_text = font.render('Game Over! Press R to restart', True, WHITE)
            text_rect = game_over_text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
            pygame.draw.rect(screen, BLACK, text_rect.inflate(20, 20))
            screen.blit(game_over_text, text_rect)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

def main():
    print("Q-Learning Snake")
    print("=" * 40)
    print("1. Train new agent")
    print("2. Watch trained agent (demo)")
    print("=" * 40)
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        demo_interval_choice = input("Watch demo every N episodes? (enter number or 0 for no demos): ").strip()
        try:
            demo_interval = int(demo_interval_choice)
            if demo_interval <= 0:
                demo_interval = None
        except:
            demo_interval = None
        
        train_agent(demo_interval)
        
        demo_choice = input("\nWatch final demo? (y/n): ").strip().lower()
        if demo_choice == 'y':
            demo_agent()
    
    elif choice == '2':
        demo_agent()
    
    else:
        print("Invalid choice")

if __name__ == '__main__':
    main()