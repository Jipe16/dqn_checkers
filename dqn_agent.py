import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Step 1: Define the Neural Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Step 2: Create the Agent
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = 1.0  # Exploration-exploitation balance
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.replay_buffer = []
        self.batch_size = 32

        # Initialize the model
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state, valid_actions_count):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(valid_actions_count))  # Random action within the valid range
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()  # Best action (exploitation)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)  # Add batch dimension

            # Calculate target Q-value
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            # Predict Q-values for the current state
            q_values = self.model(state_tensor)
            current_q = q_values[0, action]

            # Calculate loss
            target_tensor = torch.tensor([target], dtype=torch.float32).view_as(current_q)
            loss = self.loss_fn(current_q, target_tensor)

            # Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Testing the DQN Agent Initialization
if __name__ == "__main__":
    state_size = 64  # Assuming the board is represented as a 1D array of 64 elements
    action_size = 64  # Assuming the action space includes every possible move on the board
    agent = DQNAgent(state_size, action_size)

    # Test the agent's get_action method
    test_state = [0] * state_size  # A test state representing an empty board
    action = agent.get_action(test_state, 10)  # Test with 10 valid actions
    print(f"Chosen action: {action}")  # Should be a valid action index between 0 and 9

    # Test epsilon decay
    print(f"Initial epsilon: {agent.epsilon}")
    agent.update_epsilon()
    print(f"Epsilon after decay: {agent.epsilon}")