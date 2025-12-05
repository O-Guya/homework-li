import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


def orthogonal_init(layer, gain=1.0):
    """Orthogonal initialization for better gradient flow"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class ActorNetwork(nn.Module):
    """Policy network that outputs action probabilities"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Orthogonal initialization
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3, gain=0.01)  # Small gain for policy output

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def get_action_and_log_prob(self, state):
        """Sample action and return log probability"""
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(self, state, action):
        """Evaluate log probability and entropy for given state-action pairs"""
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """Value network that estimates V(s)"""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Orthogonal initialization
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPOAgent:
    """
    Independent PPO (IPPO) agent implementation with:
    - PPO-Clip objective
    - Generalized Advantage Estimation (GAE)
    - Advantage normalization
    - Gradient clipping
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        k_epochs=4,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        device='cpu'
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr_actor: Learning rate for actor network
            lr_critic: Learning rate for critic network
            gamma: Discount factor
            gae_lambda: Lambda for GAE
            clip_epsilon: Clipping parameter for PPO
            k_epochs: Number of update iterations per batch
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Max norm for gradient clipping
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Rollout buffer
        self.reset_buffer()

    def reset_buffer(self):
        """Clear the rollout buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def select_action(self, state, is_training=True):
        """
        Select action using current policy

        Args:
            state: Current state
            is_training: If True, sample from policy; if False, use deterministic (argmax)

        Returns:
            action: Selected action (int)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.actor(state_tensor)

            if is_training:
                # Sample from policy
                dist = Categorical(logits=logits)
                action = dist.sample()
            else:
                # Deterministic: choose action with highest probability
                action = torch.argmax(logits, dim=-1)

        return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the rollout buffer

        Note: Unlike Q-Learning, we don't store next_state here because
        PPO uses on-policy rollout. We compute values during storage.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)

        with torch.no_grad():
            # Get log probability and value for this transition
            log_prob, _ = self.actor.evaluate_actions(state_tensor, action_tensor)
            value = self.critic(state_tensor)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        self.dones.append(done)

    def compute_gae(self, next_value=0.0):
        """
        Compute Generalized Advantage Estimation (GAE)

        Args:
            next_value: Value of the next state (0 if terminal)

        Returns:
            advantages: Computed advantages
            returns: Discounted returns (targets for value function)
        """
        advantages = []
        gae = 0

        # Iterate backwards through the episode
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = self.values[t + 1]

            # TD error: ´_t = r_t + ³V(s_{t+1}) - V(s_t)
            delta = self.rewards[t] + self.gamma * next_value_t * (1 - self.dones[t]) - self.values[t]

            # GAE: A_t = ´_t + (³»)´_{t+1} + (³»)²´_{t+2} + ...
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(self.values, dtype=np.float32)

        return advantages, returns

    def update(self, batch_size=None):
        """
        Update policy and value networks using PPO

        Args:
            batch_size: Not used in this implementation (kept for interface compatibility)

        Returns:
            Dictionary containing loss information
        """
        if len(self.states) == 0:
            return None

        # Compute GAE
        # For the last state, we need to bootstrap the value
        # If episode ended, next_value = 0; otherwise compute from last state
        if self.dones[-1]:
            next_value = 0.0
        else:
            with torch.no_grad():
                last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
                next_value = self.critic(last_state).item()

        advantages, returns = self.compute_gae(next_value)

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_tensor = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO update for K epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for epoch in range(self.k_epochs):
            # Evaluate current policy
            log_probs, entropy = self.actor.evaluate_actions(states_tensor, actions_tensor)
            values = self.critic(states_tensor).squeeze()

            # Compute ratio (À_¸ / À_¸_old)
            ratio = torch.exp(log_probs - old_log_probs_tensor)

            # Compute surrogate losses
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor

            # Actor loss (PPO-Clip objective + entropy bonus)
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            # Critic loss (MSE between predicted value and return)
            critic_loss = F.mse_loss(values, returns_tensor)

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            # Accumulate losses for logging
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()

        # Clear buffer after update
        self.reset_buffer()

        return {
            'actor_loss': total_actor_loss / self.k_epochs,
            'critic_loss': total_critic_loss / self.k_epochs,
            'entropy': total_entropy / self.k_epochs
        }

    def save_model(self, filepath):
        """Save model parameters"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)

    def load_model(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
