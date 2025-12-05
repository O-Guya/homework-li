import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # define the network layers
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
 
    def forward(self, x):
        """
        define forward propagation
        """
        # using ReLU activation function for the first two layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # return the output layer
        return self.fc3(x)

class QLearningAgent:
    """
    Implements a Q-learning agent
    """
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, device='cpu'):
        """
        state_dim: dimension of state space
        action_dim: dimension of action space
        lr: learning rate
        gamma: discount factor, 0 < gamma <= 1, a higher value makes the agent consider future rewards more
        epsilon: exploration rate, 0 <= epsilon < 1, a higher value makes the agent explore more
        epsilon_decay: decay rate for epsilon after each episode
        epsilon_min: minimum value for epsilon
        device: device to run the model on ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = device
        
        # initialize the Q-network and remove it to GPU if available
        self.q_net = QNetwork(state_dim, action_dim).to(device)
        # initialize the optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        # define the loss function, using Mean Squared Error
        self.criterion = nn.MSELoss()

        # Replay Buffer
        self.memory = deque(maxlen=10000)

    def select_action(self, state, is_training=True):
        """
        Selects an action using epsilon-greedy policy
        state: current state
        is_training: whether the agent is in training mode
        """
        # if random number is less than epsilon, select a random action (exploration)
        if is_training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        # otherwise, select the action with the highest Q-value (exploitation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer
        """
        self.memory.append((state, action, reward, next_state, done))

    def update(self, batch_size=64):
        """
        Update the Q-network using a batch of experiences from the replay buffer
        """
        if len(self.memory) < batch_size:
            return None  # not enough samples to update
        
        # sample a batch of experiences
        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # compute current Q values
        current_q_values = self.q_net(states).gather(1, actions)

        # calculate target Q values
        with torch.no_grad():
            next_q_values = self.q_net(next_states).max(1)[0].unsqueeze(1)

            # bellman equation, Q(s, a) = r + gamma * max_a' Q(s', a')
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))


        # backpropagation
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()