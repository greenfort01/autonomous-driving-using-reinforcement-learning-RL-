import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image
import time
import pandas as pd
import seaborn as sns
from datetime import datetime
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
        # Input shape is (channels, height, width)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        self.fc_input_size = self._get_conv_output(input_shape)
        
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def _get_conv_output(self, shape):
        # Dummy forward pass to calculate output size
        o = F.relu(self.conv1(torch.zeros(1, *shape)))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_shape, action_space):
        # Initialize DQN
        self.state_shape = state_shape  # (channels, height, width)
        self.action_space = action_space
        self.num_actions = 5  # Discretized actions: [straight, left, right, accelerate, brake]
        
        # Create networks
        self.policy_net = DQN(state_shape, self.num_actions)
        self.target_net = DQN(state_shape, self.num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Learning parameters
        self.learning_rate = 0.0001
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor
        
        # Training counters
        self.target_update_frequency = 10
        self.num_episodes = 0
        
        # New: Metrics tracking
        self.q_values_history = []
        self.action_distribution = [0] * self.num_actions
        self.total_training_steps = 0
    
    def select_action(self, state, evaluation=False):
        """Select action with epsilon-greedy policy for training, or greedy policy for evaluation"""
        if not evaluation and random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                
                # Track q-values during training (not during evaluation)
                if not evaluation:
                    self.q_values_history.append(q_values.mean().item())
                
                action = torch.argmax(q_values).item()
        
        # Track action distribution
        if not evaluation:
            self.action_distribution[action] += 1
            
        return action
    
    def action_to_env_action(self, action_idx):
        """Convert discrete action index to environment action"""
        # Map to [steering, gas, brake]
        if action_idx == 0:  # Straight
            return [0.0, 0.5, 0.0]
        elif action_idx == 1:  # Left
            return [-0.8, 0.5, 0.0]
        elif action_idx == 2:  # Right
            return [0.8, 0.5, 0.0]
        elif action_idx == 3:  # Accelerate
            return [0.0, 1.0, 0.0]
        elif action_idx == 4:  # Brake
            return [0.0, 0.0, 0.8]
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.total_training_steps += 1
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Get next Q values from target net
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
        
        # Calculate target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Calculate loss and optimize
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def get_action_distribution(self):
        """Get normalized action distribution"""
        total = sum(self.action_distribution)
        if total == 0:
            return [0] * self.num_actions
        return [count / total for count in self.action_distribution]

class MetricsTracker:
    """Class for tracking detailed training and evaluation metrics"""
    def __init__(self, log_dir="metrics"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.epsilon_values = []
        
        # Rolling window metrics
        self.window_size = 10
        
        # Convergence metrics
        self.convergence_threshold = 0.9  # When agent achieves 90% of best observed perf
        self.best_reward = -float('inf')
        self.convergence_episode = None
        
        # Evaluation metrics across environments
        self.eval_environments = []
        self.eval_rewards = {}
        self.eval_completion_rates = {}
        
        # Time metrics
        self.start_time = None
        self.training_times = []
    
    def start_training(self):
        """Mark the start of training"""
        self.start_time = time.time()
    
    def log_episode(self, episode, reward, steps, loss, epsilon):
        """Log metrics for a training episode"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(steps)
        self.episode_losses.append(loss)
        self.epsilon_values.append(epsilon)
        
        # Check for convergence
        if reward > self.best_reward:
            self.best_reward = reward
        if self.convergence_episode is None and reward >= self.convergence_threshold * self.best_reward:
            self.convergence_episode = episode
        
        # Log training time for this episode
        if self.start_time is not None:
            self.training_times.append(time.time() - self.start_time)
    
    def add_eval_environment(self, env_name):
        """Add an evaluation environment to track"""
        if env_name not in self.eval_environments:
            self.eval_environments.append(env_name)
            self.eval_rewards[env_name] = []
            self.eval_completion_rates[env_name] = []
    
    def log_evaluation(self, env_name, rewards, completion_rate):
        """Log evaluation metrics for a specific environment"""
        self.eval_rewards[env_name].append(rewards)
        self.eval_completion_rates[env_name].append(completion_rate)
    
    def get_rolling_reward(self):
        """Calculate rolling average reward"""
        if len(self.episode_rewards) < self.window_size:
            return np.mean(self.episode_rewards) if self.episode_rewards else 0
        return np.mean(self.episode_rewards[-self.window_size:])
    
    def get_convergence_info(self):
        """Get convergence information"""
        return {
            'best_reward': self.best_reward,
            'convergence_episode': self.convergence_episode,
            'convergence_percentage': self.convergence_episode / len(self.episode_rewards) * 100 if self.convergence_episode else None
        }
    
    def save_metrics_to_csv(self):
        """Save all metrics to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save training metrics
        training_df = pd.DataFrame({
            'episode': range(1, len(self.episode_rewards) + 1),
            'reward': self.episode_rewards,
            'steps': self.episode_lengths,
            'loss': self.episode_losses,
            'epsilon': self.epsilon_values,
            'time': self.training_times if len(self.training_times) == len(self.episode_rewards) else [None] * len(self.episode_rewards)
        })
        training_df.to_csv(f"{self.log_dir}/training_metrics_{timestamp}.csv", index=False)
        
        # Save evaluation metrics for each environment
        for env_name in self.eval_environments:
            eval_df = pd.DataFrame({
                'environment': [env_name] * len(self.eval_rewards[env_name]),
                'avg_reward': [np.mean(rewards) for rewards in self.eval_rewards[env_name]],
                'max_reward': [np.max(rewards) for rewards in self.eval_rewards[env_name]],
                'min_reward': [np.min(rewards) for rewards in self.eval_rewards[env_name]],
                'std_reward': [np.std(rewards) for rewards in self.eval_rewards[env_name]],
                'completion_rate': self.eval_completion_rates[env_name]
            })
            eval_df.to_csv(f"{self.log_dir}/evaluation_{env_name}_{timestamp}.csv", index=False)
    
    def plot_all_metrics(self, agent=None):
        """Generate comprehensive plots for all tracked metrics"""
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        grid = plt.GridSpec(3, 3, figure=fig)
        
        # 1. Training Rewards
        ax1 = fig.add_subplot(grid[0, 0:2])
        ax1.plot(self.episode_rewards, label='Episode Reward')
        
        # Add rolling average
        if len(self.episode_rewards) >= self.window_size:
            rolling_rewards = [np.mean(self.episode_rewards[max(0, i-self.window_size+1):i+1]) 
                            for i in range(len(self.episode_rewards))]
            ax1.plot(rolling_rewards, 'r-', label=f'{self.window_size}-Episode Rolling Avg')
        
        # Mark convergence if detected
        if self.convergence_episode is not None:
            ax1.axvline(x=self.convergence_episode, color='g', linestyle='--', 
                      label=f'Convergence at Episode {self.convergence_episode}')
        
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Training Loss
        ax2 = fig.add_subplot(grid[0, 2])
        ax2.plot(self.episode_losses)
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Loss')
        ax2.grid(True)
        
        # 3. Episode Length (Steps)
        ax3 = fig.add_subplot(grid[1, 0])
        ax3.plot(self.episode_lengths)
        ax3.set_title('Episode Length')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.grid(True)
        
        # 4. Exploration Rate (Epsilon)
        ax4 = fig.add_subplot(grid[1, 1])
        ax4.plot(self.epsilon_values)
        ax4.set_title('Exploration Rate (Epsilon)')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.grid(True)
        
        # 5. Training Time per Episode
        if self.training_times and len(self.training_times) == len(self.episode_rewards):
            ax5 = fig.add_subplot(grid[1, 2])
            cumulative_times = np.cumsum(self.training_times)
            ax5.plot(cumulative_times / 60)  # Convert to minutes
            ax5.set_title('Cumulative Training Time')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Time (minutes)')
            ax5.grid(True)
        
        # 6. Action Distribution
        if agent is not None:
            ax6 = fig.add_subplot(grid[2, 0])
            action_dist = agent.get_action_distribution()
            action_names = ['Straight', 'Left', 'Right', 'Accelerate', 'Brake']
            ax6.bar(action_names, action_dist)
            ax6.set_title('Action Distribution')
            ax6.set_ylabel('Frequency')
            plt.xticks(rotation=45)
            ax6.grid(True)
        
        # 7. Q-Values Progression
        if agent is not None and agent.q_values_history:
            ax7 = fig.add_subplot(grid[2, 1])
            # Downsample if too many points
            q_values = agent.q_values_history
            if len(q_values) > 1000:
                indices = np.linspace(0, len(q_values)-1, 1000, dtype=int)
                q_values = [q_values[i] for i in indices]
            ax7.plot(q_values)
            ax7.set_title('Q-Values Progression')
            ax7.set_xlabel('Training Step (sampled)')
            ax7.set_ylabel('Average Q-Value')
            ax7.grid(True)
        
        # 8. Evaluation Performance Across Environments
        if self.eval_environments:
            ax8 = fig.add_subplot(grid[2, 2])
            
            env_data = []
            for env_name in self.eval_environments:
                for rewards in self.eval_rewards[env_name]:
                    for r in rewards:
                        env_data.append({'Environment': env_name, 'Reward': r})
            
            if env_data:
                eval_df = pd.DataFrame(env_data)
                sns.boxplot(x='Environment', y='Reward', data=eval_df, ax=ax8)
                ax8.set_title('Evaluation Performance')
                ax8.set_ylabel('Reward')
                plt.xticks(rotation=45)
                ax8.grid(True)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{self.log_dir}/performance_metrics_{timestamp}.png", dpi=300)
        plt.close()
        
        return f"{self.log_dir}/performance_metrics_{timestamp}.png"

def preprocess_state(observation):
    """Convert observation to grayscale and resize"""
    # Convert RGB to grayscale
    gray = np.mean(observation, axis=2)
    # Resize to 84x84
    resized = np.array(Image.fromarray(gray.astype(np.uint8)).resize((84, 84)))
    # Normalize
    normalized = resized / 255.0
    # Reshape to (1, 84, 84) - add channel dimension
    return normalized.reshape(1, 84, 84)

def train(env, agent, metrics_tracker, num_episodes=500, max_steps=1000):
    """Train the agent for a number of episodes with enhanced metrics tracking"""
    metrics_tracker.start_training()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        
        for step in range(max_steps):
            # Select action
            action_idx = agent.select_action(state)
            action = agent.action_to_env_action(action_idx)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess_state(next_state)
            
            # Custom reward shaping
            # Penalize collisions and going off-track
            if reward < -0.1:  # Car going off-track or hitting obstacles
                shaped_reward = -10
            else:
                # Reward for speed and staying on track
                shaped_reward = reward * 10
            
            # Store experience
            agent.store_experience(state, action_idx, shaped_reward, next_state, done)
            
            # Learn from experiences
            loss = agent.learn()
            if loss is not None:
                episode_loss += loss
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += shaped_reward
            step_count += 1
            
            if done or step == max_steps - 1:
                break
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Update target network periodically
        if episode % agent.target_update_frequency == 0:
            agent.update_target_net()
        
        # Log metrics for this episode
        avg_loss = episode_loss / step_count if step_count > 0 else 0
        metrics_tracker.log_episode(episode, episode_reward, step_count, avg_loss, agent.epsilon)
        
        # Track progress
        rolling_reward = metrics_tracker.get_rolling_reward()
        
        print(f"Episode {episode+1}/{num_episodes}, Steps: {step_count}, "
              f"Reward: {episode_reward:.2f}, Rolling Reward: {rolling_reward:.2f}, "
              f"Epsilon: {agent.epsilon:.3f}, Loss: {avg_loss:.4f}")
        
        agent.num_episodes += 1
    
    # Save final metrics
    metrics_tracker.save_metrics_to_csv()
    
    return metrics_tracker

def evaluate(env, agent, metrics_tracker, env_name="default", num_episodes=10, render=True):
    """Evaluate the trained agent with detailed metrics"""
    all_rewards = []
    completion_count = 0
    
    metrics_tracker.add_eval_environment(env_name)
    
    for episode in range(num_episodes):
        state, info = env.reset()
        state = preprocess_state(state)
        episode_reward = 0
        step_count = 0
        
        done = False
        while not done and step_count < 1000:  # Add max steps to prevent infinite loops
            if render:
                env.render()
                time.sleep(0.01)  # Slow down rendering
            
            # Select best action (no exploration)
            action_idx = agent.select_action(state, evaluation=True)
            action = agent.action_to_env_action(action_idx)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = preprocess_state(next_state)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            step_count += 1
        
        all_rewards.append(episode_reward)
        
        # Track if the episode was completed successfully (not terminated early)
        if not terminated and step_count >= 1000:
            completion_count += 1
        
        print(f"Evaluation on {env_name} - Episode {episode+1}, Reward: {episode_reward:.2f}, Steps: {step_count}")
    
    # Calculate completion rate
    completion_rate = completion_count / num_episodes
    
    # Log evaluation metrics
    metrics_tracker.log_evaluation(env_name, all_rewards, completion_rate)
    
    # Print summary statistics
    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    min_reward = np.min(all_rewards)
    max_reward = np.max(all_rewards)
    
    print(f"\nEvaluation Summary for {env_name}:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min/Max Reward: {min_reward:.2f} / {max_reward:.2f}")
    print(f"Completion Rate: {completion_rate:.2%}")
    
    return all_rewards, completion_rate

def plot_results(metrics_tracker, agent, show_plot=True):
    """Plot comprehensive training results"""
    image_path = metrics_tracker.plot_all_metrics(agent)
    
    if show_plot:
        img = plt.imread(image_path)
        plt.figure(figsize=(18, 14))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    return image_path

def save_agent(agent, filename):
    """Save the agent's model to a file"""
    torch.save({
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'num_episodes': agent.num_episodes,
        'action_distribution': agent.action_distribution,
        'q_values_history': agent.q_values_history[-1000:] if len(agent.q_values_history) > 1000 else agent.q_values_history,
        'total_training_steps': agent.total_training_steps
    }, filename)
    print(f"Agent saved to {filename}")

def load_agent(filename, state_shape, action_space):
    """Load an agent from a file"""
    agent = DQNAgent(state_shape, action_space)
    checkpoint = torch.load(filename)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    agent.num_episodes = checkpoint['num_episodes']
    
    # Load additional metrics if available
    if 'action_distribution' in checkpoint:
        agent.action_distribution = checkpoint['action_distribution']
    if 'q_values_history' in checkpoint:
        agent.q_values_history = checkpoint['q_values_history']
    if 'total_training_steps' in checkpoint:
        agent.total_training_steps = checkpoint['total_training_steps']
    
    print(f"Agent loaded from {filename}")
    return agent

def evaluate_across_environments(agent, env_configs, metrics_tracker, num_episodes=5):
    """Evaluate the agent across multiple environment configurations"""
    results = {}
    
    for name, config in env_configs.items():
        print(f"\nEvaluating on environment: {name}")
        try:
            env = gym.make('CarRacing-v2', **config)
            rewards, completion_rate = evaluate(env, agent, metrics_tracker, env_name=name, 
                                               num_episodes=num_episodes, render=True)
            results[name] = {
                'avg_reward': np.mean(rewards),
                'completion_rate': completion_rate
            }
            env.close()
        except Exception as e:
            print(f"Error evaluating on {name}: {e}")
    
    return results

def main():
    # Create logs directory
    log_dir = "racing_metrics_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(log_dir=log_dir)
    
    # Create environment with appropriate render mode
    env = gym.make('CarRacing-v2', render_mode="human")
    state_shape = (1, 84, 84)  # Channels, height, width
    
    # Create agent
    agent = DQNAgent(state_shape, env.action_space)
    
    # Training phase
    print("Starting training...")
    metrics_tracker = train(env, agent, metrics_tracker, num_episodes=100, max_steps=1000)
    
    # Plot results
    plot_results(metrics_tracker, agent)
    
    # Save the agent
    save_agent(agent, f"{log_dir}/autonomous_driving_agent.pth")
    
    # Print convergence information
    conv_info = metrics_tracker.get_convergence_info()
    print("\nTraining Convergence Information:")
    print(f"Best Reward Achieved: {conv_info['best_reward']:.2f}")
    if conv_info['convergence_episode']:
        print(f"Converged at Episode: {conv_info['convergence_episode']} ({conv_info['convergence_percentage']:.2f}% of training)")
    else:
        print("Model did not converge according to the threshold criteria")
    
    # Evaluation phase across multiple environments
    print("\nStarting evaluations across different environments...")
    
    # Define different environment configurations to test robustness
    env_configs = {
        "Default": {"render_mode": "human"},
        "Stochastic": {"render_mode": "human", "domain_randomize": True},
        "Continuous Track": {"render_mode": "human", "continuous": True}
    }
    
    # Evaluate across environments
    eval_results = evaluate_across_environments(agent, env_configs, metrics_tracker, num_episodes=3)
    
    # Print cross-environment evaluation summary
    print("\nCross-Environment Evaluation Summary:")
    for env_name, metrics in eval_results.items():
        print(f"{env_name}: Avg Reward = {metrics['avg_reward']:.2f}, Completion Rate = {metrics['completion_rate']:.2%}")
    
    # Close the environment
    env.close()
    
    print(f"\nAll metrics and results saved to {log_dir}")

if __name__ == "__main__":
    main()
