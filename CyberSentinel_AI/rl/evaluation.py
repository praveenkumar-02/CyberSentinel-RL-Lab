"""
CyberSentinel RL Lab - Rigorous Evaluation and Benchmarking Suite
Evaluates trained reinforcement learning agents on separate, isolated episodes
WITHOUT modifying Q-tables, transition counts, value estimates, or learned policies.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from .environment import NetworkSecurityEnv


class RuleBasedBaselineAgent:
    """
    Expert heuristic / rule-based baseline agent:
    - Benign states (0, 4, 8) -> MONITOR (0)
    - Suspicious states (1, 2, 5, 6) -> INVESTIGATE (1)
    - Severe attacks (3, 7, 9, 10, 11) -> BLOCK (2)
    """

    def __init__(self):
        self.name = "Rule-Based Baseline"

    def select_action(self, state: int, evaluate: bool = True) -> int:
        if state in (0, 4, 8):
            return 0  # MONITOR
        elif state in (1, 2, 5, 6):
            return 1  # INVESTIGATE
        else:
            return 2  # BLOCK


def evaluate_agent(
    agent: Any,
    env: NetworkSecurityEnv,
    num_episodes: int = 50,
    seed: int = 1234
) -> Dict[str, Any]:
    """
    Strict evaluation harness:
    Runs separate seeded evaluation episodes with NO learning/updating of the agent.
    Computes cumulative rewards, task success, missed threats, false alarm blocks, and episode lengths.
    """
    eval_rng = np.random.default_rng(seed)

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    successful_episodes: int = 0
    missed_threats_count: int = 0
    unnecessary_blocks_count: int = 0
    total_steps_evaluated: int = 0

    for ep in range(num_episodes):
        ep_seed = int(eval_rng.integers(1, 1000000))
        state = env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0
        steps = 0
        breached = False

        while not done:
            # Strictly greedy action selection; no exploratory actions or model updates
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = env.step(action)

            ep_reward += reward
            steps += 1
            total_steps_evaluated += 1

            if info.get("early_terminal", False):
                breached = True

            # Tracking security metrics
            is_benign = (state in (0, 4, 8))
            is_severe = (state in (3, 7, 9, 10, 11))

            if is_benign and action == 2:
                unnecessary_blocks_count += 1

            if is_severe and action == 0:
                missed_threats_count += 1

            state = next_state

        if not breached:
            successful_episodes += 1

        episode_rewards.append(round(ep_reward, 2))
        episode_lengths.append(steps)

    avg_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    success_rate = (successful_episodes / num_episodes) * 100.0
    missed_threat_rate = (missed_threats_count / max(1, total_steps_evaluated)) * 100.0
    unnecessary_block_rate = (unnecessary_blocks_count / max(1, total_steps_evaluated)) * 100.0
    avg_length = float(np.mean(episode_lengths))

    agent_name = getattr(agent, "name", agent.__class__.__name__)
    if "QLearning" in agent_name:
        agent_name = "Q-Learning"
    elif "SARSA" in agent_name:
        agent_name = "SARSA"
    elif "ModelBased" in agent_name:
        agent_name = "Model-Based RL"

    return {
        "agent": agent_name,
        "num_episodes": num_episodes,
        "average_reward": round(avg_reward, 2),
        "reward_std": round(std_reward, 2),
        "min_reward": round(float(np.min(episode_rewards)), 2),
        "max_reward": round(float(np.max(episode_rewards)), 2),
        "success_rate_percent": round(success_rate, 1),
        "missed_threat_rate_percent": round(missed_threat_rate, 2),
        "unnecessary_block_rate_percent": round(unnecessary_block_rate, 2),
        "average_episode_length": round(avg_length, 1),
        "total_evaluation_steps": total_steps_evaluated,
        "evaluation_rewards": episode_rewards
    }


def compare_all_algorithms(
    env: NetworkSecurityEnv,
    train_episodes: int = 400,
    eval_episodes: int = 50,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Train and benchmark all three algorithms alongside the rule-based baseline
    under identical training interaction budgets and identical evaluation scenarios.
    """
    from .q_learning import QLearningAgent
    from .sarsa import SARSAAgent
    from .model_based import ModelBasedRLAgent

    results = []

    # 1. Rule-Based Baseline (no training required)
    baseline = RuleBasedBaselineAgent()
    eval_baseline = evaluate_agent(baseline, env, num_episodes=eval_episodes, seed=seed)
    eval_baseline["training_interactions"] = 0
    eval_baseline["planning_updates"] = 0
    eval_baseline["paradigm"] = "Heuristic Expert Rules"
    results.append(eval_baseline)

    # 2. Q-Learning
    ql = QLearningAgent(alpha=0.10, gamma=0.95, epsilon=0.25, epsilon_decay=0.995, seed=seed)
    train_res_ql = ql.train(env, num_episodes=train_episodes)
    eval_ql = evaluate_agent(ql, env, num_episodes=eval_episodes, seed=seed)
    eval_ql["training_interactions"] = train_res_ql["total_interactions"]
    eval_ql["planning_updates"] = 0
    eval_ql["paradigm"] = "Model-Free Off-Policy TD"
    results.append(eval_ql)

    # 3. SARSA
    sarsa = SARSAAgent(alpha=0.10, gamma=0.95, epsilon=0.25, epsilon_decay=0.995, seed=seed)
    train_res_sarsa = sarsa.train(env, num_episodes=train_episodes)
    eval_sarsa = evaluate_agent(sarsa, env, num_episodes=eval_episodes, seed=seed)
    eval_sarsa["training_interactions"] = train_res_sarsa["total_interactions"]
    eval_sarsa["planning_updates"] = 0
    eval_sarsa["paradigm"] = "Model-Free On-Policy TD"
    results.append(eval_sarsa)

    # 4. Model-Based RL (Learned Model + Value Iteration)
    mb = ModelBasedRLAgent(gamma=0.95, epsilon=0.25, epsilon_decay=0.995, planning_iterations=15, seed=seed)
    train_res_mb = mb.train(env, num_episodes=train_episodes)
    eval_mb = evaluate_agent(mb, env, num_episodes=eval_episodes, seed=seed)
    eval_mb["training_interactions"] = train_res_mb["total_interactions"]
    eval_mb["planning_updates"] = train_res_mb["total_planning_sweeps"]
    eval_mb["paradigm"] = "Model-Based (Learned MDP + Value Iteration)"
    results.append(eval_mb)

    return {
        "benchmark_results": results,
        "train_episodes_budget": train_episodes,
        "eval_episodes_count": eval_episodes,
        "evaluation_seed": seed,
        "summary": "Comparison conducted on identical evaluation episodes under deterministic random seeds without evaluation-time learning."
    }
