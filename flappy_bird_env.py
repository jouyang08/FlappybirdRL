"""
FlappyBirdRL - Custom Flappy Bird Environment
Copyright (c) 2025 Jackson Ouyang. All rights reserved.

This file implements a custom Flappy Bird game environment using Pygame
for training deep reinforcement learning agents. Features realistic physics,
collision detection, and state extraction for AI training.

Author: Jackson Ouyang
Email: jacksonouyang1@gmail.com
GitHub: https://github.com/jouyang08/FlappybirdRL

Permission is hereby granted to use this code for educational and research purposes.
Commercial use requires explicit permission from the author.
"""

import pygame
import random
import numpy as np
import cv2

class FlappyBirdEnv:
    def __init__(self, screen_width=400, screen_height=600):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Flappy Bird RL")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.YELLOW = (255, 255, 0)
        
        # Game parameters
        self.gravity = 0.5
        self.jump_strength = -10
        self.pipe_width = 70
        self.pipe_gap = 150
        self.pipe_speed = 3
        
        # Bird parameters
        self.bird_size = 30
        self.bird_x = 100
        
        # Clock for FPS
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        self.reset()
    
    def reset(self):
        """Reset the game environment"""
        self.bird_y = self.screen_height // 2
        self.bird_velocity = 0
        self.score = 0
        self.pipes = []
        self.game_over = False
        
        # Create initial pipes
        for i in range(3):
            pipe_x = self.screen_width + i * 200
            pipe_height = random.randint(100, self.screen_height - self.pipe_gap - 100)
            self.pipes.append({'x': pipe_x, 'height': pipe_height})
        
        return self.get_state()
    
    def step(self, action):
        """Take a step in the environment
        action: 0 = do nothing, 1 = jump
        """
        reward = 0.1  # Small positive reward for staying alive
        
        # Handle bird jump
        if action == 1:
            self.bird_velocity = self.jump_strength
        
        # Apply gravity
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity
        
        # Move pipes
        for pipe in self.pipes:
            pipe['x'] -= self.pipe_speed
        
        # Remove pipes that have gone off screen and add new ones
        if self.pipes[0]['x'] < -self.pipe_width:
            self.pipes.pop(0)
            pipe_x = self.pipes[-1]['x'] + 200
            pipe_height = random.randint(100, self.screen_height - self.pipe_gap - 100)
            self.pipes.append({'x': pipe_x, 'height': pipe_height})
        
        # Check for scoring (bird passed a pipe)
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width < self.bird_x and pipe['x'] + self.pipe_width > self.bird_x - self.pipe_speed:
                self.score += 1
                reward = 10  # Reward for passing through a pipe
        
        # Check for collisions
        if self.check_collision():
            self.game_over = True
            reward = -100  # Large negative reward for dying
        
        # Check if bird went out of bounds
        if self.bird_y < 0 or self.bird_y > self.screen_height:
            self.game_over = True
            reward = -100
        
        return self.get_state(), reward, self.game_over, {'score': self.score}
    
    def check_collision(self):
        """Check if bird collides with pipes"""
        bird_rect = pygame.Rect(self.bird_x - self.bird_size//2, self.bird_y - self.bird_size//2, 
                               self.bird_size, self.bird_size)
        
        for pipe in self.pipes:
            # Upper pipe
            upper_pipe_rect = pygame.Rect(pipe['x'], 0, self.pipe_width, pipe['height'])
            # Lower pipe
            lower_pipe_rect = pygame.Rect(pipe['x'], pipe['height'] + self.pipe_gap, 
                                        self.pipe_width, self.screen_height - pipe['height'] - self.pipe_gap)
            
            if bird_rect.colliderect(upper_pipe_rect) or bird_rect.colliderect(lower_pipe_rect):
                return True
        
        return False
    
    def get_state(self):
        """Get the current state representation for the RL agent"""
        # Find the next pipe
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width > self.bird_x:
                next_pipe = pipe
                break
        
        if next_pipe is None:
            next_pipe = self.pipes[0]
        
        # State features:
        # 1. Bird's vertical position (normalized)
        # 2. Bird's velocity (normalized)
        # 3. Horizontal distance to next pipe (normalized)
        # 4. Vertical distance to next pipe's gap center (normalized)
        # 5. Vertical distance to next pipe's top (normalized)
        # 6. Vertical distance to next pipe's bottom (normalized)
        
        bird_y_norm = self.bird_y / self.screen_height
        bird_velocity_norm = self.bird_velocity / 20  # Normalize velocity
        
        horizontal_dist = (next_pipe['x'] - self.bird_x) / self.screen_width
        gap_center = next_pipe['height'] + self.pipe_gap / 2
        vertical_dist_to_gap = (gap_center - self.bird_y) / self.screen_height
        vertical_dist_to_top = (next_pipe['height'] - self.bird_y) / self.screen_height
        vertical_dist_to_bottom = ((next_pipe['height'] + self.pipe_gap) - self.bird_y) / self.screen_height
        
        state = np.array([
            bird_y_norm,
            bird_velocity_norm,
            horizontal_dist,
            vertical_dist_to_gap,
            vertical_dist_to_top,
            vertical_dist_to_bottom
        ], dtype=np.float32)
        
        return state
    
    def render(self):
        """Render the game"""
        self.screen.fill(self.WHITE)
        
        # Draw pipes
        for pipe in self.pipes:
            # Upper pipe
            pygame.draw.rect(self.screen, self.GREEN, 
                           (pipe['x'], 0, self.pipe_width, pipe['height']))
            # Lower pipe
            pygame.draw.rect(self.screen, self.GREEN, 
                           (pipe['x'], pipe['height'] + self.pipe_gap, 
                            self.pipe_width, self.screen_height - pipe['height'] - self.pipe_gap))
        
        # Draw bird
        pygame.draw.circle(self.screen, self.YELLOW, 
                         (int(self.bird_x), int(self.bird_y)), self.bird_size//2)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, self.BLACK)
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def close(self):
        """Close the game"""
        pygame.quit()
    
    def get_screen_array(self):
        """Get the screen as a numpy array for neural network input"""
        screen_array = pygame.surfarray.array3d(self.screen)
        screen_array = np.transpose(screen_array, (1, 0, 2))
        screen_array = cv2.cvtColor(screen_array, cv2.COLOR_RGB2GRAY)
        screen_array = cv2.resize(screen_array, (84, 84))
        return screen_array / 255.0  # Normalize to [0, 1]
