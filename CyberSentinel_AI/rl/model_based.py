"""
CyberSentinel RL Lab - Model-Based Reinforcement Learning Implementation
Learned Transition Model + Empirical Reward Model + Value Iteration Planning.

Theoretical Foundation:
Model-Based RL separates learning into two distinct phases:
1. Model Learning (System Identification):
   The agent gathers real observations (s, a, r, s', done) and estimates the
   underlying Markov Decision Process transition dynamics:
   P_hat(s' | s, a) = Count(s, a, s') / N(s, a)
   R_hat(s, a, s') = SumReward(s, a, s') / Count(s, a, s')

2. Internal Planning (Value Iteration):
   Using its learned internal model, the agent performs dynamic programming
   updates via the Bellman Optimality Equation:
   V(s) <- max_a sum_s' P_hat(s' | s, a) * [ R_hat(s, a, s') + gamma * V(s') * (1 - Terminal(s, a, s')) ]

The policy is derived from the planned values without requiring millions of real environment interactions.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .environment import NetworkSecurityEnv


class ModelBasedRLAgent:
    """
    Model-Based RL Agent with learned transition dynamics and Value Iteration planning.
    """

    def __init__(
        self,
        num_states: int = NetworkSecurityEnv.NUM_STATES,
        num_actions: int = NetworkSecurityEnv.NUM_ACTIONS,
        gamma: float = 0.95,
        epsilon: float = 0.20,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.02,
        planning_iterations: int = 15,
        seed: Optional[int] = None
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.initial_epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.min_epsilon = float(min_epsilon)
        self.planning_iterations = int(planning_iterations)
        self.rng = np.random.default_rng(seed)

        # -------------------------------------------------------------
        # Learned Model Data Structures
        # -------------------------------------------------------------
        # N(s, a): total visits to state s taking action a
        self.visit_counts = np.zeros((self.num_states, self.num_actions), dtype=np.int64)
        # C(s, a, s'): count of times transition s -> a resulted in s'
        self.transition_counts = np.zeros(
            (self.num_states, self.num_actions, self.num_states), dtype=np.int64
        )
        # R_sum(s, a, s'): cumulative reward observed for transition s, a -> s'
        self.reward_sum = np.zeros(
            (self.num_states, self.num_actions, self.num_states), dtype=np.float64
        )
        # Terminal count for transition s, a -> s'
        self.terminal_counts = np.zeros(
            (self.num_states, self.num_actions, self.num_states), dtype=np.int64
        )

        # Value function V(s) and policy pi(s)
        self.v_table = np.zeros(self.num_states, dtype=np.float64)
        self.policy = np.zeros(self.num_states, dtype=np.int64)
        # Action-value table Q(s, a) computed during planning
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

        # Training history records
        self.history: Dict[str, List[Any]] = {
            "episode_rewards": [],
            "moving_avg_rewards": [],
            "epsilon_history": [],
            "episode_lengths": [],
            "planning_deltas": []
        }

    def record_observation(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> None:
        """
        Model Learning Step: Update transition counts and empirical rewards.
        """
        self.visit_counts[state, action] += 1
        self.transition_counts[state, action, next_state] += 1
        self.reward_sum[state, action, next_state] += reward
        if done:
            self.terminal_counts[state, action, next_state] += 1

    def get_transition_probabilities(self, state: int, action: int) -> np.ndarray:
        """
        Calculate empirical transition probability distribution P_hat(s' | s, a).
        If unvisited, returns uniform distribution over all states.
        """
        total = self.visit_counts[state, action]
        if total == 0:
            # Unobserved transition: uniform prior
            return np.full(self.num_states, 1.0 / self.num_states, dtype=np.float64)
        return self.transition_counts[state, action] / float(total)

    def get_expected_reward(self, state: int, action: int, next_state: int) -> float:
        """
        Retrieve empirical expected reward R_hat(s, a, s').
        """
        c = self.transition_counts[state, action, next_state]
        if c == 0:
            return 0.0
        return float(self.reward_sum[state, action, next_state] / c)

    def is_transition_terminal(self, state: int, action: int, next_state: int) -> bool:
        """
        Check if transition was observed as terminal.
        """
        c = self.transition_counts[state, action, next_state]
        if c == 0:
            return False
        return (self.terminal_counts[state, action, next_state] / c) >= 0.5

    def plan_value_iteration(self) -> float:
        """
        Internal Planning: Run Value Iteration using the learned model.
        Updates self.v_table, self.q_table, and self.policy.
        Returns the maximum Bellman delta observed during the final sweep.
        """
        max_delta = 0.0

        for _ in range(self.planning_iterations):
            delta = 0.0
            new_v = np.copy(self.v_table)

            for s in range(self.num_states):
                action_values = np.zeros(self.num_actions, dtype=np.float64)

                for a in range(self.num_actions):
                    if self.visit_counts[s, a] == 0:
                        # Unvisited action: neutral default value
                        action_values[a] = 0.0
                        continue

                    # Expected value using learned transition probabilities
                    p_vec = self.transition_counts[s, a] / float(self.visit_counts[s, a])
                    ev = 0.0

                    for s_next in np.flatnonzero(p_vec > 0):
                        p = p_vec[s_next]
                        r = self.get_expected_reward(s, a, s_next)
                        is_term = self.is_transition_terminal(s, a, s_next)
                        future = 0.0 if is_term else self.gamma * self.v_table[s_next]
                        ev += p * (r + future)

                    action_values[a] = ev

                self.q_table[s] = action_values
                best_val = np.max(action_values)
                delta = max(delta, abs(new_v[s] - best_val))
                new_v[s] = best_val
                self.policy[s] = int(np.argmax(action_values))

            self.v_table = new_v
            max_delta = delta

        return max_delta

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """
        Select action: Epsilon-greedy exploration with respect to the planned policy.
        If evaluate=True, strictly follow the planned policy.
        """
        if not evaluate and (self.rng.random() < self.epsilon):
            return int(self.rng.choice(self.num_actions))

        # Check if any action is unvisited for this state; prior to planning, explore unvisited
        unvisited = np.flatnonzero(self.visit_counts[state] == 0)
        if not evaluate and len(unvisited) > 0 and self.rng.random() < 0.30:
            return int(self.rng.choice(unvisited))

        return int(self.policy[state])

    def decay_epsilon(self) -> None:
        """
        Decay exploration rate per episode until reaching min_epsilon.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, env: NetworkSecurityEnv, num_episodes: int = 500) -> Dict[str, Any]:
        """
        Train agent across multiple episodes:
        Interleaves real environment steps (experience gathering) with
        internal planning steps (Value Iteration).
        """
        ep_rewards = []
        ep_lengths = []
        eps_list = []
        deltas = []

        for ep in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = self.select_action(state, evaluate=False)
                next_state, reward, done, _ = env.step(action)

                # 1. Model learning: record observation into transition dynamics
                self.record_observation(state, action, reward, next_state, done)

                total_reward += reward
                steps += 1
                state = next_state

            # 2. Internal planning: update value function & policy using learned model
            plan_delta = self.plan_value_iteration()
            deltas.append(round(plan_delta, 4))

            self.decay_epsilon()

            ep_rewards.append(round(total_reward, 2))
            ep_lengths.append(steps)
            eps_list.append(round(self.epsilon, 4))

        # Moving averages
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
        self.history["planning_deltas"].extend(deltas)

        # Calculate model coverage: how many (s, a) pairs have been visited
        observed_pairs = int(np.count_nonzero(self.visit_counts))
        total_pairs = self.num_states * self.num_actions
        coverage_pct = round((observed_pairs / total_pairs) * 100, 1)

        return {
            "algorithm": "Model-Based RL",
            "episodes_trained": num_episodes,
            "final_epsilon": round(self.epsilon, 4),
            "final_moving_avg": moving_avgs[-1] if moving_avgs else 0.0,
            "best_episode_reward": float(np.max(ep_rewards)) if ep_rewards else 0.0,
            "total_interactions": int(np.sum(ep_lengths)),
            "model_coverage_percent": coverage_pct,
            "total_planning_sweeps": num_episodes * self.planning_iterations
        }

    def get_policy(self) -> Dict[int, Dict[str, Any]]:
        """
        Extract planned policy pi(s), value function V(s), and model statistics.
        """
        policy_dict = {}
        for s in range(self.num_states):
            decoded = NetworkSecurityEnv.decode_state(s)
            best_a = int(self.policy[s])
            policy_dict[s] = {
                "state_id": s,
                "description": decoded["description"],
                "category": decoded["category"],
                "state_value_v": round(float(self.v_table[s]), 3),
                "q_values": [round(float(q), 3) for q in self.q_table[s]],
                "recommended_action": best_a,
                "recommended_action_name": NetworkSecurityEnv.ACTIONS[best_a],
                "total_visits": int(np.sum(self.visit_counts[s])),
                "action_visits": [int(v) for v in self.visit_counts[s]]
            }
        return policy_dict

    def get_transition_matrix(self, action: int) -> Dict[str, Any]:
        """
        Return the learned transition probability matrix P(s' | s, a) for a specific action.
        """
        matrix = []
        for s in range(self.num_states):
            p_vec = self.get_transition_probabilities(s, action)
            matrix.append([round(float(p), 3) for p in p_vec])
        return {
            "action": action,
            "action_name": NetworkSecurityEnv.ACTIONS[action],
            "matrix": matrix
        }

    def reset_agent(self) -> None:
        """
        Reset model, counts, value tables, and history back to initial state.
        """
        self.visit_counts.fill(0)
        self.transition_counts.fill(0)
        self.reward_sum.fill(0.0)
        self.terminal_counts.fill(0)
        self.v_table.fill(0.0)
        self.q_table.fill(0.0)
        self.policy.fill(0)
        self.epsilon = self.initial_epsilon
        for k in self.history:
            self.history[k] = []
