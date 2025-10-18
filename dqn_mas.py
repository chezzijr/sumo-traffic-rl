"""Multi-Agent DQN Training Script for 3x3 Traffic Grid.

This script implements coordinated multi-agent reinforcement learning where:
- Each intersection has its own DQN agent (9 agents total)
- Agents share a global experience replay buffer
- Agents observe neighbor states for coordination
- Uses PettingZoo parallel_env for true multi-agent RL
"""

import argparse
import os
import sys
from collections import deque
from typing import Dict, List, Tuple, Any
import random

import numpy as np
import pandas as pd

from agents.dqn import DQNAgent, ReplayBuffer


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import parallel_env
from sumo_rl.exploration import EpsilonGreedy


# 3x3 Grid Topology: Define which intersections are neighbors
# Grid layout:
# J1 - J2 - J3
# |    |    |
# J4 - J5 - J6
# |    |    |
# J7 - J8 - J9
GRID_TOPOLOGY = {
    "J1": ["J2", "J4"],
    "J2": ["J1", "J3", "J5"],
    "J3": ["J2", "J6"],
    "J4": ["J1", "J5", "J7"],
    "J5": ["J2", "J4", "J6", "J8"],  # Center, has 4 neighbors
    "J6": ["J3", "J5", "J9"],
    "J7": ["J4", "J8"],
    "J8": ["J5", "J7", "J9"],
    "J9": ["J6", "J8"],
}


class MultiAgentCoordinator:
    """Coordinates multiple DQN agents with shared experience replay."""

    def __init__(
        self,
        env: Any,
        alpha: float = 0.001,
        gamma: float = 0.95,
        initial_epsilon: float = 0.05,
        min_epsilon: float = 0.005,
        decay: float = 1.0,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        use_neighbor_obs: bool = True,
    ):
        """Initialize multi-agent coordinator.

        Args:
            env: PettingZoo parallel environment
            alpha: Learning rate for DQN
            gamma: Discount factor
            initial_epsilon: Starting epsilon for exploration
            min_epsilon: Minimum epsilon value
            decay: Epsilon decay rate
            buffer_size: Size of shared replay buffer
            batch_size: Mini-batch size for training
            target_update_freq: Frequency of target network updates
            use_neighbor_obs: Whether to include neighbor observations
        """
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.use_neighbor_obs = use_neighbor_obs

        # Get list of traffic signal IDs from environment
        # PettingZoo uses 'agents' attribute
        self.ts_ids = None
        self.agents = {}
        self.shared_replay_buffer = ReplayBuffer(buffer_size)

        # Exploration strategy (shared across all agents)
        self.exploration = EpsilonGreedy(
            initial_epsilon=initial_epsilon, min_epsilon=min_epsilon, decay=decay
        )

        # Performance tracking
        self.episode_rewards = {ts: [] for ts in GRID_TOPOLOGY.keys()}
        self.system_metrics = []

        # Store hyperparameters for agent initialization
        self.alpha = alpha
        self.target_update_freq = target_update_freq

        # Maximum local observation dimension (detected from environment)
        # Different junctions have different obs dimensions: J5=27D, J2/J4/J6/J8=23D, corners=19D
        self.max_local_obs_dim = None

    def initialize_agents(self, initial_observations: Dict[str, Any]):
        """Initialize DQN agents after environment reset.

        Args:
            initial_observations: Initial observations from env.reset()
        """
        self.ts_ids = list(initial_observations.keys())

        # Detect maximum local observation dimension across all agents
        self.max_local_obs_dim = max(
            len(np.array(obs)) for obs in initial_observations.values()
        )
        print(f"Max local observation dimension detected: {self.max_local_obs_dim}")

        for ts in self.ts_ids:
            # Get observation space from environment
            obs_space = self.env.observation_space(ts)
            action_space = self.env.action_space(ts)

            # Augment state with neighbor observations if enabled
            initial_state = self._augment_observation(ts, initial_observations)

            self.agents[ts] = DQNAgent(
                starting_state=initial_state,
                state_space=obs_space,
                action_space=action_space,
                alpha=self.alpha,
                gamma=self.gamma,
                exploration_strategy=self.exploration,
                buffer_size=0,  # Don't use individual buffers
                batch_size=self.batch_size,
                target_update_freq=self.target_update_freq,
            )

            # Override individual replay buffer - use shared one
            self.agents[ts].replay_buffer = self.shared_replay_buffer

            # Initialize episode rewards
            if ts not in self.episode_rewards:
                self.episode_rewards[ts] = []

    def _augment_observation(
        self, ts_id: str, observations: Dict[str, Any]
    ) -> np.ndarray:
        """Augment local observation with neighbor states.

        Args:
            ts_id: Traffic signal ID
            observations: Dictionary of all observations

        Returns:
            Augmented observation array with fixed dimension
        """
        local_obs = np.array(observations[ts_id])

        # CRITICAL: Pad local observation to maximum dimension
        # Different junctions have different obs sizes (19D, 23D, 27D)
        # All must be padded to max (27D) for shared replay buffer compatibility
        if self.max_local_obs_dim is not None:
            current_local_dim = len(local_obs)
            if current_local_dim < self.max_local_obs_dim:
                # Pad with zeros to match maximum dimension
                padding = np.zeros(self.max_local_obs_dim - current_local_dim)
                local_obs = np.concatenate([local_obs, padding])

        # If not using neighbor observations, return padded local obs
        if not self.use_neighbor_obs:
            return local_obs

        # Maximum number of neighbors in the grid (J5 has 4 neighbors)
        MAX_NEIGHBORS = 4
        FEATURES_PER_NEIGHBOR = 4  # Queue features extracted per neighbor

        # Get neighbor observations
        neighbors = GRID_TOPOLOGY.get(ts_id, [])
        neighbor_features = []

        for neighbor_id in neighbors:
            if neighbor_id in observations:
                neighbor_obs = observations[neighbor_id]
                # Extract only queue-related features (first few dimensions)
                # Assuming queue lengths are in first part of observation
                if isinstance(neighbor_obs, np.ndarray):
                    # Take first 4 features (queues) from each neighbor
                    queue_features = neighbor_obs[:FEATURES_PER_NEIGHBOR]
                    neighbor_features.extend(queue_features)

        # Pad to fixed size: all agents should have same observation dimension
        # This ensures compatibility with shared replay buffer
        expected_neighbor_size = MAX_NEIGHBORS * FEATURES_PER_NEIGHBOR
        current_size = len(neighbor_features)

        if current_size < expected_neighbor_size:
            # Pad with zeros for missing neighbors
            padding = [0.0] * (expected_neighbor_size - current_size)
            neighbor_features.extend(padding)

        # Concatenate padded local observation with padded neighbor features
        augmented_obs = np.concatenate([local_obs, neighbor_features])

        return augmented_obs

    def select_actions(self, observations: Dict[str, Any]) -> Dict[str, int]:
        """Select actions for all agents.

        Args:
            observations: Current observations for all agents

        Returns:
            Dictionary of actions for each agent
        """
        actions = {}
        for ts in self.ts_ids:
            # Update agent's state with augmented observation
            self.agents[ts].state = self._augment_observation(ts, observations)
            # Select action
            actions[ts] = self.agents[ts].act()

        return actions

    def store_transitions(
        self,
        observations: Dict[str, Any],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_observations: Dict[str, Any],
        dones: Dict[str, bool],
    ):
        """Store transitions for all agents in shared buffer.

        Args:
            observations: Current observations
            actions: Actions taken
            rewards: Rewards received
            next_observations: Next observations
            dones: Done flags
        """
        for ts in self.ts_ids:
            state = self._augment_observation(ts, observations)
            next_state = self._augment_observation(ts, next_observations)

            self.shared_replay_buffer.push(
                state=state,
                action=actions[ts],
                reward=rewards[ts],
                next_state=next_state,
                done=dones.get(ts, False),
            )

    def train_agents(self):
        """Train all agents using shared experience replay."""
        if len(self.shared_replay_buffer) < self.batch_size:
            return

        # Each agent learns from shared buffer
        for ts in self.ts_ids:
            # Sample from shared buffer
            states, actions, rewards, next_states, dones = (
                self.shared_replay_buffer.sample(self.batch_size)
            )

            # Update agent's Q-network
            # We manually perform the learning step without storing in individual buffer
            agent = self.agents[ts]

            # Convert to tensors
            import torch

            state_batch = torch.FloatTensor(
                np.array([agent._state_to_tensor(s).cpu().numpy() for s in states])
            ).to(agent.device)
            action_batch = torch.LongTensor(actions).to(agent.device)
            reward_batch = torch.FloatTensor(rewards).to(agent.device)
            next_state_batch = torch.FloatTensor(
                np.array([agent._state_to_tensor(s).cpu().numpy() for s in next_states])
            ).to(agent.device)
            done_batch = torch.FloatTensor(dones).to(agent.device)

            # Current Q-values
            current_q_values = agent.q_network(state_batch).gather(
                1, action_batch.unsqueeze(1)
            ).squeeze(1)

            # Target Q-values
            with torch.no_grad():
                next_q_values = agent.target_network(next_state_batch).max(1)[0]
                target_q_values = reward_batch + (1 - done_batch) * agent.gamma * next_q_values

            # Compute loss
            loss = agent.criterion(current_q_values, target_q_values)

            # Optimize
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

            # Update target network
            agent.learn_step_counter += 1
            if agent.learn_step_counter % agent.target_update_freq == 0:
                agent.target_network.load_state_dict(agent.q_network.state_dict())

    def update_metrics(self, rewards: Dict[str, float]):
        """Update performance metrics.

        Args:
            rewards: Rewards received by each agent
        """
        for ts in self.ts_ids:
            self.agents[ts].acc_reward += rewards[ts]

    def get_system_metrics(self) -> Dict[str, float]:
        """Calculate system-wide performance metrics.

        Returns:
            Dictionary of system metrics
        """
        metrics = {
            "total_reward": sum(agent.acc_reward for agent in self.agents.values()),
            "avg_reward": np.mean([agent.acc_reward for agent in self.agents.values()]),
            "min_reward": min(agent.acc_reward for agent in self.agents.values()),
            "max_reward": max(agent.acc_reward for agent in self.agents.values()),
            "replay_buffer_size": len(self.shared_replay_buffer),
        }
        return metrics

    def reset_episode_rewards(self):
        """Reset accumulated rewards for new episode."""
        for agent in self.agents.values():
            agent.acc_reward = 0


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent DQN for Traffic Control")
    parser.add_argument("--alpha", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Initial epsilon")
    parser.add_argument("--min-epsilon", type=float, default=0.005, help="Minimum epsilon")
    parser.add_argument("--decay", type=float, default=0.9999, help="Epsilon decay")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--target-update", type=int, default=1000, help="Target update frequency")
    parser.add_argument("--neighbor-obs", action="store_true", help="Use neighbor observations")
    parser.add_argument("--gui", action="store_true", help="Use SUMO GUI")
    parser.add_argument("--output-dir", type=str, default="outputs/marl", help="Output directory")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    for run in range(1, args.runs + 1):
        print(f"\n{'='*60}")
        print(f"Starting Run {run}/{args.runs}")
        print(f"{'='*60}")

        # Create parallel environment for multi-agent RL
        env = parallel_env(
            net_file="scenarios/3x3/vn.net.xml",
            route_file="scenarios/3x3/vn.rou.xml",
            use_gui=args.gui,
            num_seconds=80000,
            min_green=5,
            delta_time=5,
            sumo_warnings=False,
        )

        # Initialize multi-agent coordinator
        coordinator = MultiAgentCoordinator(
            env=env,
            alpha=args.alpha,
            gamma=args.gamma,
            initial_epsilon=args.epsilon,
            min_epsilon=args.min_epsilon,
            decay=args.decay,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            target_update_freq=args.target_update,
            use_neighbor_obs=args.neighbor_obs,
        )

        for episode in range(1, args.episodes + 1):
            print(f"\nEpisode {episode}/{args.episodes}")

            # Reset environment
            observations, infos = env.reset()

            # Initialize agents on first episode
            if episode == 1:
                coordinator.initialize_agents(observations)
            else:
                # Reset episode rewards
                coordinator.reset_episode_rewards()
                # Update states for all agents
                for ts in coordinator.ts_ids:
                    coordinator.agents[ts].state = coordinator._augment_observation(
                        ts, observations
                    )

            step = 0
            terminated = {agent: False for agent in env.agents}
            truncated = {agent: False for agent in env.agents}

            while not (all(terminated.values()) or all(truncated.values())):
                # Select actions for all agents
                actions = coordinator.select_actions(observations)

                # Store previous observations
                prev_observations = observations

                # Execute actions in environment
                observations, rewards, terminated, truncated, infos = env.step(actions)

                # Store transitions in shared buffer
                coordinator.store_transitions(
                    prev_observations, actions, rewards, observations, terminated
                )

                # Train all agents from shared buffer
                coordinator.train_agents()

                # Update metrics
                coordinator.update_metrics(rewards)

                step += 1

                # Print progress every 1000 steps
                if step % 1000 == 0:
                    metrics = coordinator.get_system_metrics()
                    print(
                        f"  Step {step}: Avg Reward: {metrics['avg_reward']:.2f}, "
                        f"Buffer: {metrics['replay_buffer_size']}"
                    )

            # Episode finished
            metrics = coordinator.get_system_metrics()
            print(f"\nEpisode {episode} Summary:")
            print(f"  Total System Reward: {metrics['total_reward']:.2f}")
            print(f"  Average Agent Reward: {metrics['avg_reward']:.2f}")
            print(f"  Min/Max Agent Reward: {metrics['min_reward']:.2f} / {metrics['max_reward']:.2f}")
            print(f"  Replay Buffer Size: {metrics['replay_buffer_size']}")

            # Save episode data
            env.save_csv(f"{args.output_dir}/run{run}", episode)

        env.close()
        print(f"\nRun {run} completed!")

    print(f"\n{'='*60}")
    print("All runs completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
