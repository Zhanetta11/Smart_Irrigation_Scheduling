"""
Standard Deep Q-Network (DQN)

Components:
- Q-Network
- Experience Replay
- Target Network
- Epsilon-Greedy Exploration
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Neural network approximating action values."""

    def __init__(self, state_size=5, action_size=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Store and sample past experiences."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32).copy(),
            bool(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Standard DQN agent with a target network."""

    def __init__(
        self,
        state_size=5,
        action_size=4,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_size=10000,
        target_update_frequency=10,
        seed=42
    ):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.action_size = action_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        self.policy_net = QNetwork(
            state_size, action_size
        ).to(self.device)

        self.target_net = QNetwork(
            state_size, action_size
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(buffer_size)

        self.training_steps = 0

        print("DQN Agent initialized")
        print("Device:", self.device)

    def select_action(self, state, training=True):
        """Choose an action using epsilon-greedy exploration."""

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(
            state, action, reward, next_state, done
        )

    def train_step(self):
        """Perform one gradient update."""

        if len(self.memory) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(self.batch_size)

        states = torch.as_tensor(
            states, device=self.device
        )

        actions = torch.as_tensor(
            actions, device=self.device
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            rewards, device=self.device
        )

        next_states = torch.as_tensor(
            next_states, device=self.device
        )

        dones = torch.as_tensor(
            dones, device=self.device
        )

        # Predicted Q-values for selected actions
        current_q = self.policy_net(states).gather(
            1, actions
        ).squeeze(1)

        # Bellman target
        with torch.no_grad():
            next_q = self.target_net(
                next_states
            ).max(dim=1).values

            target_q = (
                rewards
                + self.gamma * next_q * (1.0 - dones)
            )

        loss = self.loss_function(
            current_q, target_q
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        self.training_steps += 1

        # Update target network periodically
        if (
            self.training_steps
            % self.target_update_frequency == 0
        ):
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        return float(loss.item())

    def decay_epsilon(self):
        """Reduce exploration after each episode."""

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def save(self, path):
        """Save trained model and exploration state."""

        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps
        }, path)

    def load(self, path):
        """Load a trusted model checkpoint."""

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )

        self.policy_net.load_state_dict(
            checkpoint["policy_state_dict"]
        )

        self.target_net.load_state_dict(
            checkpoint["target_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]