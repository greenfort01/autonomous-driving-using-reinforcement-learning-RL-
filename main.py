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
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()
    
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
        self.optimizer.step()
        
        return loss.item()

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

def train(env, agent, num_episodes=500, max_steps=1000):
    """Train the agent for a number of episodes"""
    rewards_history = []
    losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        episode_reward = 0
        episode_loss = 0
        
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
            
            if done or step == max_steps - 1:
                break
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Update target network periodically
        if episode % agent.target_update_frequency == 0:
            agent.update_target_net()
        
        # Track progress
        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history)
        avg_loss = episode_loss / (step + 1) if step > 0 else 0
        losses.append(avg_loss)
        
        print(f"Episode {episode+1}/{num_episodes}, Steps: {step+1}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        agent.num_episodes += 1
    
    return rewards_history, losses

def evaluate(env, agent, num_episodes=5, render=True):
    """Evaluate the trained agent"""
    for episode in range(num_episodes):
        state, info = env.reset()  # Properly unpack the return value
        state = preprocess_state(state)
        episode_reward = 0
        
        done = False
        while not done:
            if render:
                env.render()
                time.sleep(0.01)  # Slow down rendering
            
            # Select best action (no exploration)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent.policy_net(state_tensor)
                action_idx = torch.argmax(q_values).item()
            
            action = agent.action_to_env_action(action_idx)
            
            # Take action - fix to handle new Gym API
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated  # Combine terminated and truncated flags
            next_state = preprocess_state(next_state)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        print(f"Evaluation Episode {episode+1}, Reward: {episode_reward:.2f}")
def plot_results(rewards, losses):
    """Plot training results"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def save_agent(agent, filename):
    """Save the agent's model to a file"""
    torch.save({
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'num_episodes': agent.num_episodes,
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
    print(f"Agent loaded from {filename}")
    return agent

def main():
    # Create environment with appropriate render mode
    # Use 'human' for visualization during training and evaluation
    env = gym.make('CarRacing-v2', render_mode="human")
    state_shape = (1, 84, 84)  # Channels, height, width
    
    # Create agent
    agent = DQNAgent(state_shape, env.action_space)
    
    # Training phase
    print("Starting training...")
    rewards, losses = train(env, agent, num_episodes=100, max_steps=1000)
    
    # Plot results
    plot_results(rewards, losses)
    
    # Save the agent
    save_agent(agent, "autonomous_driving_agent.pth")
    
    # Evaluation phase
    print("Starting evaluation...")
    # Create a new environment for evaluation with rendering
    eval_env = gym.make('CarRacing-v2', render_mode="human")
    evaluate(eval_env, agent, num_episodes=3)
    
    # Close environments
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()