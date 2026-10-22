"""
CyberSentinel RL Lab - Q-Learning Algorithm Implementation
Off-Policy Temporal Difference (TD) Control.

Theoretical Foundation:
Q-learning is an off-policy temporal difference control algorithm.
It updates state-action values using the Bellman optimality operator:
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

Why Off-Policy?
It is off-policy because the target value r + gamma * max_a' Q(s', a') assumes the greedy
optimal action is executed in the next state, irrespective of the action actually selected
by the epsilon-greedy behavior policy.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .environment import NetworkSecurityEnv


class QLearningAgent:
    """
    Tabular Q-Learning Agent for discrete network threat response.
    """

    def __init__(
        self,
        num_states: int = NetworkSecurityEnv.NUM_STATES,
        num_actions: int = NetworkSecurityEnv.NUM_ACTIONS,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon: float = 0.20,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.02,
        seed: Optional[int] = None
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.initial_epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.min_epsilon = float(min_epsilon)
        self.rng = np.random.default_rng(seed)

        # Tabular Q-Table: shape (num_states, num_actions)
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        # Tracking visit counts per (state, action)
        self.visit_counts = np.zeros((self.num_states, self.num_actions), dtype=np.int64)

        # Training history records
        self.history: Dict[str, List[Any]] = {
            "episode_rewards": [],
            "moving_avg_rewards": [],
            "epsilon_history": [],
            "episode_lengths": [],
            "td_errors": []
        }

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """
        Select an action using an epsilon-greedy behavior policy.
        If evaluate=True, strictly select the greedy action (no exploration).
        """
        if not evaluate and (self.rng.random() < self.epsilon):
            # Exploration: choose random action uniformly
            return int(self.rng.choice(self.num_actions))

        # Exploitation: break ties randomly among actions with maximum Q-value
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_q, atol=1e-7))
        return int(self.rng.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """
        Execute one off-policy Q-Learning TD update.
        Target = r + gamma * max_a' Q(s', a')  (if not done)
        Target = r                             (if done)
        """
        self.visit_counts[state, action] += 1
        current_q = self.q_table[state, action]

        if done:
            target = reward
        else:
            # Off-policy: max over all possible actions in s', regardless of behavior policy
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * max_next_q

        td_error = float(target - current_q)
        self.q_table[state, action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self) -> None:
        """
        Decay exploration rate per episode until reaching min_epsilon.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, env: NetworkSecurityEnv, num_episodes: int = 500) -> Dict[str, Any]:
        """
        Train agent across multiple episodes in the environment.
        """
        ep_rewards = []
        ep_lengths = []
        eps_list = []
        avg_td_errors = []

        for ep in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            ep_td_errors = []

            while not done:
                action = self.select_action(state, evaluate=False)
                next_state, reward, done, _ = env.step(action)
                td_err = self.update(state, action, reward, next_state, done)
                ep_td_errors.append(abs(td_err))

                total_reward += reward
                steps += 1
                state = next_state

            self.decay_epsilon()

            ep_rewards.append(round(total_reward, 2))
            ep_lengths.append(steps)
            eps_list.append(round(self.epsilon, 4))
            avg_td_errors.append(round(float(np.mean(ep_td_errors)) if ep_td_errors else 0.0, 4))

        # Calculate moving average rewards (window=10)
        window = 10
        moving_avgs = []
        for i in range(len(ep_rewards)):
            start_idx = max(0, i - window + 1)
            subset = ep_rewards[start_idx : i + 1]
            moving_avgs.append(round(float(np.mean(subset)), 2))

        self.history["episode_rewards"].extend(ep_rewards)
        self.history["moving_avg_rewards"].extend(moving_avgs)
        self.history["epsilon_history"].extend(eps_list)
        self.history["episode_lengths"].extend(ep_lengths)
        self.history["td_errors"].extend(avg_td_errors)

        return {
            "algorithm": "Q-Learning",
            "episodes_trained": num_episodes,
            "final_epsilon": round(self.epsilon, 4),
            "final_moving_avg": moving_avgs[-1] if moving_avgs else 0.0,
            "best_episode_reward": float(np.max(ep_rewards)) if ep_rewards else 0.0,
            "total_interactions": int(np.sum(ep_lengths))
        }

    def get_policy(self) -> Dict[int, Dict[str, Any]]:
        """
        Extract greedy policy pi(s) and Q-values for all states.
        """
        policy = {}
        for s in range(self.num_states):
            q_vals = self.q_table[s]
            best_a = int(np.argmax(q_vals))
            decoded = NetworkSecurityEnv.decode_state(s)
            policy[s] = {
                "state_id": s,
                "description": decoded["description"],
                "category": decoded["category"],
                "q_values": [round(float(q), 3) for q in q_vals],
                "recommended_action": best_a,
                "recommended_action_name": NetworkSecurityEnv.ACTIONS[best_a],
                "visit_count": int(np.sum(self.visit_counts[s]))
            }
        return policy

    def reset_agent(self) -> None:
        """
        Reset Q-table, visit counts, and history back to initial state.
        """
        self.q_table.fill(0.0)
        self.visit_counts.fill(0)
        self.epsilon = self.initial_epsilon
        for k in self.history:
            self.history[k] = []
