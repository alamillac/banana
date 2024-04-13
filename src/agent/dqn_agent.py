import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import QNetwork, QNetworkVisual

BUFFER_SIZE = int(1e5)  # replay buffer size
BUFFER_SIZE_VISUAL = int(
    15000
)  # replay buffer size. Adjust this value based on the memory available
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

# Prioritized Experience Replay
UNIFORM_SAMPLING_ALPHA = 0.6
BETA_START = 0.4
BETA_INCREMENT = 0.001
PRIORITY_EPSILON = 1e-5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, visual=False, seed=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)

        # Q-Network
        self.visual = visual
        if self.visual:
            buffer_size = BUFFER_SIZE_VISUAL
            QNet = QNetworkVisual
        else:
            buffer_size = BUFFER_SIZE
            QNet = QNetwork

        self.qnetwork_local = QNet(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNet(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, BATCH_SIZE, seed)
        self.beta = BETA_START

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if random.random() < eps:
            return np.random.randint(self.action_size)  # select a random action

        # Select a greedy action
        if self.visual:
            state = torch.from_numpy(state).float().to(device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, sampling_weights, exp_idx = (
            experiences
        )

        # ------------------- compute and minimize the loss ------------------- #

        # Get the expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Get the Q values from target model
        with torch.no_grad():
            # Double DQN
            next_actions = torch.argmax(
                self.qnetwork_local(next_states), dim=1
            ).unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        buffer_size = len(self.memory)
        weights = (buffer_size * sampling_weights) ** (-self.beta) # Importance sampling weights
        weights = weights / weights.max()  # Normalize the weights

        weighted_loss = (
            weights * F.mse_loss(Q_expected, Q_targets, reduction="none")
        ).mean()

        # Update the priorities
        td_errors = torch.abs(Q_expected - Q_targets).detach().squeeze().cpu().numpy()
        self.memory.update_priority(exp_idx, td_errors + PRIORITY_EPSILON)

        # Minimize the loss
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        # ------------------- update beta ------------------- #

        self.beta = min(1.0, self.beta + BETA_INCREMENT)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, path):
        torch.save(self.qnetwork_local.state_dict(), path)

    def load(self, path):
        self.qnetwork_local.load_state_dict(torch.load(path))
        self.qnetwork_target.load_state_dict(torch.load(path))


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.memory = deque(maxlen=self.buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

        # Prioritized Experience Replay
        self.priorities = np.zeros(self.buffer_size)
        self.max_priority = 1.0
        self.next_idx = 0
        self.current_size = 0

        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

        if self.current_size < self.buffer_size:
            self.current_size += 1

        # Add priority
        self.priorities[self.next_idx] = self.max_priority
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    def update_priority(self, idx, priorities):
        self.priorities[idx] = priorities
        self.max_priority = max(np.max(priorities), self.max_priority)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        priorities = (
            self.priorities[:self.current_size]**UNIFORM_SAMPLING_ALPHA
        )  # To make the sampling more uniform and reduce overfitting
        sampling_weights = priorities / np.sum(priorities)
        idx_experiences = np.random.choice(
            range(self.current_size),
            size=self.batch_size,
            replace=False,
            p=sampling_weights,
        )  # Sample based on the priority

        idx_adjusted = self.next_idx - self.current_size + idx_experiences
        experiences = [self.memory[idx] for idx in idx_adjusted]

        sampling_weights = (
            torch.from_numpy(sampling_weights[idx_experiences]).float().to(device)
        )
        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            sampling_weights,
            idx_experiences,
        )

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
