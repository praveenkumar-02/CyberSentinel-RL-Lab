"""
CyberSentinel RL Lab - Training Manager & State Orchestrator
Thread-safe session state managing active agents, live environment stepping,
and training history tracking.
"""

import threading
from typing import Dict, Any, Optional
import numpy as np

from .environment import NetworkSecurityEnv
from .q_learning import QLearningAgent
from .sarsa import SARSAAgent
from .model_based import ModelBasedRLAgent
from .evaluation import evaluate_agent, compare_all_algorithms


class TrainingManager:
    """
    Central manager for environment interactions, training routines, and policy inspection.
    """

    def __init__(self, seed: int = 42):
        self.lock = threading.Lock()
        self.seed = seed
        self.env = NetworkSecurityEnv(max_steps=20, seed=seed)

        # Initial default agents
        self.active_algorithm = "Q-Learning"
        self.agents = {
            "Q-Learning": QLearningAgent(alpha=0.10, gamma=0.95, epsilon=0.20, seed=seed),
            "SARSA": SARSAAgent(alpha=0.10, gamma=0.95, epsilon=0.20, seed=seed),
            "Model-Based RL": ModelBasedRLAgent(gamma=0.95, epsilon=0.20, planning_iterations=15, seed=seed)
        }

        # Training state
        self.is_training = False
        self.training_progress = 0.0
        self.training_status_text = "Idle. System ready for simulation or training."

        # Quick pre-train agents slightly so initial policy viewer isn't completely empty
        self._initial_warmup()

    def _initial_warmup(self):
        """Warm up agents with a brief 50-episode baseline so initial view has learned data."""
        for name, agent in self.agents.items():
            agent.train(self.env, num_episodes=50)
        self.env.reset()

    def set_active_algorithm(self, algorithm_name: str) -> None:
        """Switch active algorithm between Q-Learning, SARSA, and Model-Based RL."""
        with self.lock:
            if algorithm_name in self.agents:
                self.active_algorithm = algorithm_name

    def get_active_agent(self) -> Any:
        return self.agents[self.active_algorithm]

    def reset_simulation(self) -> Dict[str, Any]:
        """Reset live simulation environment and return initial state."""
        with self.lock:
            state = self.env.reset()
            decoded = self.env.decode_state(state)
            return {
                "state": state,
                "decoded": decoded,
                "step": 0,
                "done": False,
                "active_algorithm": self.active_algorithm
            }

    def simulation_step(self, action: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute one step in the live simulation environment.
        If action is None, uses active agent's policy.
        """
        with self.lock:
            if self.env.terminal:
                self.env.reset()

            state_before = self.env.current_state
            agent = self.get_active_agent()

            if action is None:
                # Agent chooses action based on policy
                action = agent.select_action(state_before, evaluate=True)

            next_state, reward, done, info = self.env.step(action)

            # Retrieve agent values for current state
            if hasattr(agent, "q_table"):
                q_vals = [round(float(q), 3) for q in agent.q_table[state_before]]
            else:
                q_vals = [0.0, 0.0, 0.0]

            v_val = round(float(agent.v_table[state_before]), 3) if hasattr(agent, "v_table") else 0.0

            return {
                "state_before": state_before,
                "state_before_info": self.env.decode_state(state_before),
                "action": action,
                "action_name": self.env.ACTIONS[action],
                "reward": reward,
                "next_state": next_state,
                "next_state_info": self.env.decode_state(next_state),
                "done": done,
                "outcome": info["outcome"],
                "explanation": info["explanation"],
                "step": self.env.current_step,
                "q_values": q_vals,
                "state_value": v_val,
                "active_algorithm": self.active_algorithm
            }

    def run_one_episode(self) -> Dict[str, Any]:
        """Run an entire episode with the active agent and return the trajectory."""
        with self.lock:
            state = self.env.reset()
            agent = self.get_active_agent()
            trajectory = []
            total_reward = 0.0
            done = False

            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward

                trajectory.append({
                    "step": info["step"],
                    "state": state,
                    "state_desc": info["state_info"]["description"],
                    "action": action,
                    "action_name": info["action_name"],
                    "reward": reward,
                    "next_state": next_state,
                    "next_state_desc": info["next_state_info"]["description"],
                    "outcome": info["outcome"],
                    "done": done
                })
                state = next_state

            return {
                "algorithm": self.active_algorithm,
                "total_reward": round(total_reward, 2),
                "episode_length": len(trajectory),
                "trajectory": trajectory
            }

    def train_agent(
        self,
        algorithm: str,
        episodes: int = 300,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon: float = 0.20,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.02,
        planning_iterations: int = 15,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Train or retrain an agent with user-defined hyperparameters."""
        with self.lock:
            self.active_algorithm = algorithm

            # Reconfigure or initialize agent
            if algorithm == "Q-Learning":
                agent = QLearningAgent(
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    min_epsilon=min_epsilon,
                    seed=seed
                )
            elif algorithm == "SARSA":
                agent = SARSAAgent(
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    min_epsilon=min_epsilon,
                    seed=seed
                )
            else:  # Model-Based RL
                agent = ModelBasedRLAgent(
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                    min_epsilon=min_epsilon,
                    planning_iterations=planning_iterations,
                    seed=seed
                )

            self.agents[algorithm] = agent

            # Execute training
            summary = agent.train(self.env, num_episodes=episodes)
            self.env.reset()

            return {
                "status": "success",
                "algorithm": algorithm,
                "summary": summary,
                "history": agent.history,
                "policy": agent.get_policy()
            }

    def get_overview_stats(self) -> Dict[str, Any]:
        """Compile overall statistics for the overview dashboard."""
        with self.lock:
            agent = self.get_active_agent()
            hist = agent.history
            ep_rewards = hist.get("episode_rewards", [])
            total_eps = len(ep_rewards)
            latest_r = ep_rewards[-1] if total_eps > 0 else 0.0
            moving_avg = hist.get("moving_avg_rewards", [])
            latest_m_avg = moving_avg[-1] if len(moving_avg) > 0 else 0.0
            cur_eps = round(getattr(agent, "epsilon", 0.0), 4)

            # Quick evaluation of active agent
            eval_res = evaluate_agent(agent, self.env, num_episodes=20, seed=123)

            return {
                "active_algorithm": self.active_algorithm,
                "total_episodes_trained": total_eps,
                "latest_episode_reward": latest_r,
                "moving_average_reward": latest_m_avg,
                "current_exploration_rate": cur_eps,
                "task_success_rate": eval_res["success_rate_percent"],
                "missed_threat_rate": eval_res["missed_threat_rate_percent"],
                "unnecessary_block_rate": eval_res["unnecessary_block_rate_percent"],
                "average_eval_reward": eval_res["average_reward"],
                "total_states": NetworkSecurityEnv.NUM_STATES,
                "total_actions": NetworkSecurityEnv.NUM_ACTIONS
            }

    def reset_active_agent(self) -> Dict[str, Any]:
        """Reset the active agent's tables and training history."""
        with self.lock:
            agent = self.get_active_agent()
            agent.reset_agent()
            self.env.reset()
            return {"status": "success", "message": f"{self.active_algorithm} agent was reset."}
