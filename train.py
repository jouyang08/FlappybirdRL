"""
FlappyBirdRL - Deep Reinforcement Learning for Flappy Bird
Copyright (c) 2025 Jackson Ouyang. All rights reserved.

This file is part of FlappyBirdRL, a deep reinforcement learning system
that trains AI agents to master Flappy Bird using DQN algorithms.

Author: Jackson Ouyang
Email: jacksonouyang1@gmail.com
GitHub: https://github.com/jouyang08/FlappybirdRL

Simple trainer for the custom Flappy Bird environment
Focus on the core RL training without external dependencies
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent

def train_flappy_bird(episodes=1000, show_every=100):
    """
    Train the DQN agent on our custom Flappy Bird environment
    """
    print("🎮 FLAPPY BIRD RL TRAINING")
    print("=" * 40)
    print("Training on custom Pygame environment")
    
    # Create environment and agent
    env = FlappyBirdEnv()
    agent = DQNAgent(state_size=6, action_size=2, lr=0.001, epsilon_decay=0.995)
    
    # Training tracking
    scores = []
    best_score = 0
    episode_rewards = []
    
    print(f"Training for {episodes} episodes...")
    print(f"Showing game every {show_every} episodes")
    print("Press 'Q' to close game window when it appears")
    
    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            # Show game window periodically
            show_game = (episode % show_every == 0) or (episode < 10)
            
            while True:
                # Agent chooses action
                action = agent.act(state)
                
                # Take step in environment
                next_state, reward, done, info = env.step(action)
                
                # Store experience and train
                agent.remember(state, action, reward, next_state, done)
                if len(agent.replay_buffer) > agent.batch_size:
                    agent.replay()
                
                # Update state and tracking
                state = next_state
                total_reward += reward
                steps += 1
                
                # Render game if showing
                if show_game:
                    env.render()
                    
                    # Check for quit
                    import pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                            show_game = False
                
                if done:
                    break
                
                # Safety limit
                if steps > 1000:
                    break
            
            # Episode complete
            score = info['score']
            scores.append(score)
            episode_rewards.append(total_reward)
            
            # Save best model
            if score > best_score:
                best_score = score
                os.makedirs('models', exist_ok=True)
                agent.save('models/flappy_best.pth')
                print(f"🏆 NEW BEST SCORE: {best_score}")
            
            # Progress report
            if episode % 50 == 0 or score > best_score:
                avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
                avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
                
                print(f"Episode {episode:4d} | Score: {score:2d} | Steps: {steps:3d} | Avg Score: {avg_score:5.1f} | Avg Reward: {avg_reward:6.1f} | Best: {best_score:2d} | ε: {agent.epsilon:.3f}")
            
            # Save periodic checkpoint
            if episode > 0 and episode % 200 == 0:
                agent.save(f'models/flappy_checkpoint_{episode}.pth')
        
        env.close()
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        env.close()
    
    # Training complete
    print(f"\n📊 TRAINING RESULTS:")
    print(f"   Episodes completed: {len(scores)}")
    print(f"   Best score: {max(scores) if scores else 0}")
    print(f"   Average score: {np.mean(scores) if scores else 0:.1f}")
    print(f"   Final exploration rate: {agent.epsilon:.3f}")
    
    # Save final model
    agent.save('models/flappy_final.pth')
    print("   ✅ Final model saved!")
    
    # Plot training progress
    if len(scores) > 10:
        plot_training_results(scores, episode_rewards)
    
    return agent, scores

def plot_training_results(scores, rewards):
    """Plot training progress"""
    plt.figure(figsize=(12, 4))
    
    # Plot scores
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Scores over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # Calculate and plot moving average
    if len(scores) > 50:
        moving_avg = []
        for i in range(50, len(scores)):
            moving_avg.append(np.mean(scores[i-50:i]))
        plt.plot(range(50, len(scores)), moving_avg, 'r-', label='50-episode average')
        plt.legend()
    
    # Plot rewards
    plt.subplot(1, 2, 2)
    plt.plot(rewards)
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    print("📈 Training plots saved as 'training_progress.png'")

def play_trained_agent(model_path='models/flappy_best.pth'):
    """
    Watch a trained agent play
    """
    print("👀 WATCHING TRAINED AGENT")
    print("=" * 30)
    
    # Create environment and agent
    env = FlappyBirdEnv()
    agent = DQNAgent(state_size=6, action_size=2)
    
    # Load trained model
    try:
        agent.load(model_path)
        agent.epsilon = 0  # No exploration - pure learned behavior
        print(f"✅ Loaded model: {model_path}")
    except FileNotFoundError:
        print(f"❌ Model not found: {model_path}")
        print("Train an agent first!")
        return
    
    print("\n🎮 Watching agent play...")
    print("Press 'Q' to quit, SPACE to restart episode")
    
    try:
        episode = 1
        while True:
            print(f"\n--- Game {episode} ---")
            state = env.reset()
            steps = 0
            
            while True:
                # Agent chooses action (no exploration)
                action = agent.act(state)
                
                # Take step
                state, reward, done, info = env.step(action)
                steps += 1
                
                # Render game
                env.render()
                
                # Handle events
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        env.close()
                        return
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        print("Restarting episode...")
                        done = True
                
                if done:
                    print(f"Episode {episode} complete: Score {info['score']}, Steps {steps}")
                    episode += 1
                    break
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped watching")
    
    env.close()

def play_manually():
    """
    Play the game manually (human player)
    """
    print("🎮 MANUAL PLAY MODE")
    print("=" * 20)
    print("Press SPACE to jump, Q to quit")
    
    env = FlappyBirdEnv()
    
    try:
        while True:
            print("\n--- New Game ---")
            state = env.reset()
            
            while True:
                # Default action (no jump)
                action = 0
                
                # Handle input
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        env.close()
                        return
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        action = 1  # Jump
                
                # Take step
                state, reward, done, info = env.step(action)
                
                # Render
                env.render()
                
                if done:
                    print(f"Game Over! Final Score: {info['score']}")
                    input("Press Enter for new game...")
                    break
    
    except KeyboardInterrupt:
        print("\n⏹️ Exiting manual play")
    
    env.close()

def main():
    """Main menu"""
    print("🎮 FLAPPY BIRD RL SYSTEM")
    print("=" * 30)
    print("Custom Pygame environment with DQN agent")
    print()
    
    while True:
        print("Options:")
        print("1. Train new agent")
        print("2. Watch trained agent")
        print("3. Play manually")
        print("4. Quick test (10 episodes)")
        print("5. Exit")
        
        choice = input("\nChoose option (1-5): ").strip()
        
        if choice == '1':
            episodes = int(input("Number of episodes (default 1000): ") or "1000")
            show_freq = int(input("Show game every N episodes (default 100): ") or "100")
            train_flappy_bird(episodes, show_freq)
        
        elif choice == '2':
            model_path = input("Model path (default: models/flappy_best.pth): ").strip()
            if not model_path:
                model_path = "models/flappy_best.pth"
            play_trained_agent(model_path)
        
        elif choice == '3':
            play_manually()
        
        elif choice == '4':
            print("Quick test training...")
            train_flappy_bird(episodes=10, show_every=2)
        
        elif choice == '5':
            print("Goodbye! 🎮")
            break
        
        else:
            print("Invalid choice. Please choose 1-5.")

if __name__ == "__main__":
    main()
