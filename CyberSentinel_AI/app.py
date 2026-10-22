"""
CyberSentinel RL Lab — Intelligent Reinforcement Learning Cybersecurity Simulator
Flask Backend & Application Routing

Implements REST API endpoints for:
- Environment state, stepping, and trajectory execution
- Q-Learning, SARSA, and Model-Based Reinforcement Learning training
- Policy extraction, Q-table, and transition probability inspection
- Rigorous separate-episode evaluation and multi-algorithm benchmarking
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_file

# Add project directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rl.environment import NetworkSecurityEnv
from rl.training_manager import TrainingManager
from rl.evaluation import compare_all_algorithms

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_PATH = os.path.join(BASE_DIR, "data", "network_traffic.csv")

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# Instantiate central thread-safe Training Manager
manager = TrainingManager(seed=42)


@app.route("/")
def index():
    """Render the primary CyberSentinel RL Lab application interface."""
    return render_template("index.html")


@app.route("/api/overview", methods=["GET"])
def api_overview():
    """Retrieve global dashboard metrics and active algorithm status."""
    try:
        stats = manager.get_overview_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/env/state", methods=["GET"])
def api_env_state():
    """Retrieve current state and decoded features of the simulation environment."""
    try:
        state = manager.env.current_state
        return jsonify({
            "state_id": state,
            "decoded": manager.env.decode_state(state),
            "step": manager.env.current_step,
            "max_steps": manager.env.max_steps,
            "done": manager.env.terminal,
            "active_algorithm": manager.active_algorithm
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/env/reset", methods=["POST"])
def api_env_reset():
    """Reset the live simulation environment."""
    try:
        res = manager.reset_simulation()
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/env/step", methods=["POST"])
def api_env_step():
    """
    Execute one single step in the simulation.
    Accepts optional 'action' integer (0, 1, 2). If omitted, uses agent's greedy policy.
    """
    try:
        data = request.get_json(silent=True) or {}
        action = data.get("action")
        if action is not None:
            try:
                action = int(action)
                if action not in (0, 1, 2):
                    return jsonify({"error": "Action must be 0 (MONITOR), 1 (INVESTIGATE), or 2 (BLOCK)."}), 400
            except ValueError:
                return jsonify({"error": "Invalid action value."}), 400

        res = manager.simulation_step(action=action)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/env/run_episode", methods=["POST"])
def api_env_run_episode():
    """Run an entire simulation episode using the active agent's policy and return trajectory."""
    try:
        res = manager.run_one_episode()
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    """
    Train an agent using user-specified hyperparameters.
    Supported algorithms: 'Q-Learning', 'SARSA', 'Model-Based RL'.
    """
    try:
        data = request.get_json(silent=True) or {}
        algorithm = data.get("algorithm", "Q-Learning")
        if algorithm not in ("Q-Learning", "SARSA", "Model-Based RL"):
            return jsonify({"error": f"Unsupported algorithm '{algorithm}'. Must be Q-Learning, SARSA, or Model-Based RL."}), 400

        # Validate hyperparameters with sensible bounds
        try:
            episodes = int(data.get("episodes", 300))
            episodes = max(10, min(2000, episodes))
            alpha = float(data.get("alpha", 0.10))
            alpha = max(0.01, min(1.0, alpha))
            gamma = float(data.get("gamma", 0.95))
            gamma = max(0.50, min(0.999, gamma))
            epsilon = float(data.get("epsilon", 0.20))
            epsilon = max(0.0, min(1.0, epsilon))
            epsilon_decay = float(data.get("epsilon_decay", 0.995))
            epsilon_decay = max(0.90, min(1.0, epsilon_decay))
            min_epsilon = float(data.get("min_epsilon", 0.02))
            min_epsilon = max(0.0, min(0.50, min_epsilon))
            planning_iterations = int(data.get("planning_iterations", 15))
            planning_iterations = max(1, min(50, planning_iterations))
            seed = int(data.get("seed", 42))
        except (ValueError, TypeError) as ve:
            return jsonify({"error": f"Invalid hyperparameter format: {str(ve)}"}), 400

        res = manager.train_agent(
            algorithm=algorithm,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            planning_iterations=planning_iterations,
            seed=seed
        )
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


@app.route("/api/train/history", methods=["GET"])
def api_train_history():
    """Retrieve full training curves for the active agent."""
    try:
        agent = manager.get_active_agent()
        return jsonify({
            "algorithm": manager.active_algorithm,
            "history": agent.history,
            "total_episodes": len(agent.history.get("episode_rewards", []))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/agent/policy", methods=["GET"])
def api_agent_policy():
    """Retrieve learned policy, Q-values, and state-values for all 12 states."""
    try:
        agent = manager.get_active_agent()
        policy = agent.get_policy()
        return jsonify({
            "algorithm": manager.active_algorithm,
            "policy": policy,
            "actions": NetworkSecurityEnv.ACTIONS
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/agent/transitions", methods=["GET"])
def api_agent_transitions():
    """Retrieve learned empirical transition probability matrix for Model-Based RL."""
    try:
        action = int(request.args.get("action", 0))
        if action not in (0, 1, 2):
            action = 0

        mb_agent = manager.agents["Model-Based RL"]
        mat_info = mb_agent.get_transition_matrix(action)
        return jsonify(mat_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/agent/reset", methods=["POST"])
def api_agent_reset():
    """Reset the active agent's learned Q-table or transition model."""
    try:
        res = manager.reset_active_agent()
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """Evaluate active agent on separate, unlearned evaluation episodes."""
    try:
        data = request.get_json(silent=True) or {}
        num_episodes = int(data.get("num_episodes", 50))
        num_episodes = max(5, min(200, num_episodes))
        seed = int(data.get("seed", 1234))

        from rl.evaluation import evaluate_agent
        agent = manager.get_active_agent()
        eval_res = evaluate_agent(agent, manager.env, num_episodes=num_episodes, seed=seed)
        return jsonify(eval_res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST", "GET"])
def api_compare():
    """
    Run full benchmark comparison across all 3 RL algorithms and the Rule-Based Baseline
    under identical training interaction budgets and deterministic evaluation episodes.
    """
    try:
        data = request.get_json(silent=True) or {} if request.method == "POST" else {}
        train_episodes = int(data.get("train_episodes", 300))
        train_episodes = max(50, min(1000, train_episodes))
        eval_episodes = int(data.get("eval_episodes", 40))
        eval_episodes = max(10, min(100, eval_episodes))
        seed = int(data.get("seed", 42))

        comp_res = compare_all_algorithms(
            env=manager.env,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            seed=seed
        )
        return jsonify(comp_res)
    except Exception as e:
        return jsonify({"error": f"Benchmark comparison failed: {str(e)}"}), 500


@app.route("/api/sample")
def api_sample():
    """Download sample network traffic flow dataset."""
    if os.path.exists(DATA_PATH):
        return send_file(
            DATA_PATH,
            as_attachment=True,
            download_name="network_traffic_sample.csv",
            mimetype="text/csv"
        )
    return jsonify({"error": "Sample data file not found."}), 404


if __name__ == "__main__":
    print("=" * 70)
    print("  CyberSentinel RL Lab — Reinforcement Learning Security Engine")
    print("  Active at: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=True)
