"""Deep Q-Network (DQN) Agent for Traffic Signal Control.

This module implements a DQN agent compatible with the sumo-rl library's QLAgent interface.
It includes experience replay, target network, and neural network-based Q-value approximation.
"""

import random
from collections import deque
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity: int = 50000):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ):
        """Store a transition in the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] | None = None):
        """Initialize Q-network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions (default: [128, 128])
        """
        super(QNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        # Build network layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state: State tensor

        Returns:
            Q-values for each action
        """
        return self.network(state)


class DQNAgent:
    """Deep Q-Network agent compatible with sumo-rl QLAgent interface."""

    def __init__(
        self,
        starting_state: Any,
        state_space: Any,
        action_space: Any,
        alpha: float = 0.001,
        gamma: float = 0.95,
        exploration_strategy: Any = None,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        hidden_dims: List[int] | None = None,
        device: str | None = None,
    ):
        """Initialize DQN agent.

        Args:
            starting_state: Initial state
            state_space: State space (gymnasium.Space)
            action_space: Action space (gymnasium.Space)
            alpha: Learning rate for optimizer
            gamma: Discount factor
            exploration_strategy: Exploration strategy (e.g., EpsilonGreedy)
            buffer_size: Size of replay buffer
            batch_size: Mini-batch size for training
            target_update_freq: Frequency of target network updates (in steps)
            hidden_dims: Hidden layer dimensions for Q-network
            device: Device to run on ('cuda' or 'cpu', auto-detect if None)
        """
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.gamma = gamma
        self.exploration = exploration_strategy
        self.acc_reward = 0

        # DQN-specific parameters
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        # Determine state dimension from actual starting_state (most reliable)
        # DQN works with continuous observations, not encoded discrete states
        if isinstance(starting_state, np.ndarray):
            self.state_dim = int(np.prod(starting_state.shape))
        elif isinstance(starting_state, (list, tuple)):
            self.state_dim = len(starting_state)
        elif isinstance(starting_state, (int, np.integer, float)):
            self.state_dim = 1
        elif hasattr(state_space, "shape") and len(state_space.shape) > 0:
            # Fallback to state_space if starting_state is unavailable
            self.state_dim = int(np.prod(state_space.shape))
        elif hasattr(state_space, "n"):
            # Discrete space
            self.state_dim = state_space.n
        else:
            raise ValueError(
                f"Cannot determine state dimension from starting_state: {starting_state} "
                f"(type: {type(starting_state)}) or state_space: {state_space}"
            )

        # Action dimension
        self.action_dim = action_space.n

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize networks
        self.q_network = QNetwork(self.state_dim, self.action_dim, hidden_dims).to(
            self.device
        )
        self.target_network = QNetwork(self.state_dim, self.action_dim, hidden_dims).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)

        # Loss function
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Q-table for exploration strategy compatibility
        # This is maintained as a fallback for exploration strategies that expect it
        self.q_table = {}

    def _state_to_tensor(self, state: Any) -> torch.Tensor:
        """Convert state to tensor format.

        Args:
            state: State in any format (preferably numpy array from observation)

        Returns:
            State as tensor with shape (state_dim,)
        """
        # Convert to numpy array first
        if isinstance(state, np.ndarray):
            state_array = state.flatten().astype(np.float32)
        elif isinstance(state, (tuple, list)):
            state_array = np.array(state, dtype=np.float32).flatten()
        elif isinstance(state, (int, np.integer, float, np.floating)):
            # Single scalar value
            state_array = np.array([float(state)], dtype=np.float32)
        elif isinstance(state, torch.Tensor):
            # Already a tensor
            return state.flatten().float().to(self.device)
        else:
            raise TypeError(
                f"Unsupported state type: {type(state)}. "
                f"Expected numpy array, list, tuple, or scalar. "
                f"State value: {state}"
            )

        # Validate dimensions
        if state_array.shape[0] != self.state_dim:
            raise ValueError(
                f"State dimension mismatch: expected {self.state_dim}, "
                f"got {state_array.shape[0]}. State: {state}"
            )

        return torch.FloatTensor(state_array).to(self.device)

    def _get_q_values(self, state: Any) -> np.ndarray:
        """Get Q-values for a state using the Q-network.

        Args:
            state: Current state

        Returns:
            Array of Q-values for each action
        """
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        self.q_network.train()
        return q_values

    def _state_to_hashable(self, state: Any) -> tuple:
        """Convert state to hashable format for use as dict key.

        Args:
            state: State in any format

        Returns:
            Hashable representation (tuple)
        """
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        elif isinstance(state, (list, tuple)):
            return tuple(state)
        elif isinstance(state, (int, float, str)):
            return (state,)
        else:
            # Try to convert to tuple
            return tuple(np.array(state).flatten())

    def act(self) -> int:
        """Choose action using exploration strategy and Q-network.

        Returns:
            Selected action
        """
        # Get Q-values from network
        q_values = self._get_q_values(self.state)

        # Use exploration strategy if provided
        if self.exploration is not None:
            # Convert state to hashable for q_table compatibility
            state_key = self._state_to_hashable(self.state)

            # Update q_table with current Q-values
            self.q_table[state_key] = q_values

            # Use exploration strategy
            self.action = self.exploration.choose(
                self.q_table, state_key, self.action_space
            )
        else:
            # Greedy action selection
            self.action = int(np.argmax(q_values))

        return self.action

    def learn(self, next_state: Any, reward: float, done: bool = False):
        """Update Q-network with new experience.

        Args:
            next_state: Next state
            reward: Reward received
            done: Whether episode is done
        """
        # Store transition in replay buffer
        self.replay_buffer.push(self.state, self.action, reward, next_state, done)

        # Update state and accumulate reward
        self.state = next_state
        self.acc_reward += reward

        # Update q_table for next state (for exploration strategy compatibility)
        # Note: We only update it when exploration strategy is being used
        if self.exploration is not None:
            next_state_key = self._state_to_hashable(next_state)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = self._get_q_values(next_state)

        # Train only if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample mini-batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        state_batch = torch.FloatTensor(
            np.array([self._state_to_tensor(s).cpu().numpy() for s in states])
        ).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(
            np.array([self._state_to_tensor(s).cpu().numpy() for s in next_states])
        ).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(state_batch).gather(
            1, action_batch.unsqueeze(1)
        ).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str):
        """Save agent's networks and state.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learn_step_counter": self.learn_step_counter,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load agent's networks and state.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.learn_step_counter = checkpoint["learn_step_counter"]
