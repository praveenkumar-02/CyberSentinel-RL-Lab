"""
CyberSentinel RL Lab - Simulated Network Security Environment
A discrete, reproducible, and mathematically rigorous simulation environment
for intelligent cybersecurity intrusion response and decision-making.

State space size: 12 states (3 traffic levels x 2 login levels x 2 behavior levels)
Action space size: 3 actions (0: MONITOR, 1: INVESTIGATE, 2: BLOCK)
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class NetworkSecurityEnv:
    """
    Simulated discrete Network Security Decision Environment.
    Educational and safe: Does NOT interact with real networks, packets, or hosts.
    """

    NUM_STATES = 12
    NUM_ACTIONS = 3

    # Human-readable action names
    ACTIONS = {
        0: "MONITOR",       # Passive logging, allow flow without intervention
        1: "INVESTIGATE",   # Forensic inspection, allocate SOC analyst triage
        2: "BLOCK",         # Terminate connection and apply firewall drop rule
    }

    ACTION_DESCRIPTIONS = {
        0: "Allow connection and maintain passive telemetry monitoring.",
        1: "Assign SOC analyst to deep-dive connection payloads and credentials.",
        2: "Issue active firewall block rule to terminate the connection immediately."
    }

    # State component mappings
    TRAFFIC_LEVELS = {0: "Low", 1: "Medium", 2: "High"}
    LOGIN_LEVELS = {0: "Normal", 1: "Suspicious"}
    BEHAVIOR_LEVELS = {0: "Normal", 1: "Anomalous"}

    def __init__(self, max_steps: int = 20, seed: Optional[int] = None):
        """
        Initialize environment.
        :param max_steps: Maximum decision steps per episode.
        :param seed: Random seed for deterministic reproducibility.
        """
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.current_state = 0
        self.current_step = 0
        self.terminal = False
        self.reset(seed=seed)

    @classmethod
    def encode_state(cls, traffic: int, failed_logins: int, behavior: int) -> int:
        """
        Encode 3 sub-features into a single integer state ID in [0, 11].
        traffic in {0, 1, 2}, failed_logins in {0, 1}, behavior in {0, 1}.
        """
        return int(traffic * 4 + failed_logins * 2 + behavior)

    @classmethod
    def decode_state(cls, state_id: int) -> Dict[str, Any]:
        """
        Decode integer state ID [0, 11] into feature components and text labels.
        """
        traffic = state_id // 4
        failed_logins = (state_id % 4) // 2
        behavior = state_id % 2

        # Threat classification of the state
        is_benign = (state_id in (0, 4, 8))
        is_severe = (state_id in (3, 7, 9, 10, 11))
        is_suspicious = not is_benign and not is_severe

        if is_benign:
            category = "Benign"
        elif is_severe:
            category = "Severe Attack"
        else:
            category = "Suspicious"

        state_names = {
            0: "Low Traffic, Normal Logins, Normal Behavior (Benign Background)",
            1: "Low Traffic, Normal Logins, Anomalous Behavior (Stealth Reconnaissance / Probe)",
            2: "Low Traffic, Suspicious Logins, Normal Behavior (Targeted Brute-Force)",
            3: "Low Traffic, Suspicious Logins, Anomalous Behavior (Active Exploit Attempt)",
            4: "Medium Traffic, Normal Logins, Normal Behavior (Typical Enterprise Workload)",
            5: "Medium Traffic, Normal Logins, Anomalous Behavior (Application Vulnerability Scan)",
            6: "Medium Traffic, Suspicious Logins, Normal Behavior (Distributed Credential Stuffing)",
            7: "Medium Traffic, Suspicious Logins, Anomalous Behavior (Multi-Vector Breach Attempt)",
            8: "High Traffic, Normal Logins, Normal Behavior (Peak Legitimate Business Hours)",
            9: "High Traffic, Normal Logins, Anomalous Behavior (Volumetric DDoS Flood)",
            10: "High Traffic, Suspicious Logins, Normal Behavior (Credential Stuffing Storm)",
            11: "High Traffic, Suspicious Logins, Anomalous Behavior (Coordinated Cyber Attack)"
        }

        return {
            "state_id": state_id,
            "traffic": traffic,
            "traffic_name": cls.TRAFFIC_LEVELS[traffic],
            "failed_logins": failed_logins,
            "failed_logins_name": cls.LOGIN_LEVELS[failed_logins],
            "behavior": behavior,
            "behavior_name": cls.BEHAVIOR_LEVELS[behavior],
            "category": category,
            "description": state_names.get(state_id, f"State {state_id}")
        }

    def reset(self, seed: Optional[int] = None) -> int:
        """
        Reset the environment for a new episode.
        Initializes step count, terminal flag, and picks an initial connection state.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.terminal = False

        # Distribution over initial states: 60% benign background, 40% initial anomalies
        if self.rng.random() < 0.60:
            traffic = self.rng.choice([0, 1, 2], p=[0.40, 0.45, 0.15])
            self.current_state = self.encode_state(traffic, 0, 0)
        else:
            traffic = self.rng.choice([0, 1, 2], p=[0.30, 0.40, 0.30])
            logins = self.rng.choice([0, 1], p=[0.40, 0.60])
            behavior = self.rng.choice([0, 1], p=[0.40, 0.60])
            self.current_state = self.encode_state(traffic, logins, behavior)

        return self.current_state

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Execute one action in the environment.
        :param action: 0 (MONITOR), 1 (INVESTIGATE), 2 (BLOCK)
        :return: (next_state, reward, done, info)
        """
        if self.terminal:
            raise RuntimeError("Cannot step in a terminal environment. Please reset() first.")

        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Must be in {list(self.ACTIONS.keys())}")

        state = self.current_state
        self.current_step += 1

        is_benign = (state in (0, 4, 8))
        is_severe = (state in (3, 7, 9, 10, 11))
        is_suspicious = not is_benign and not is_severe

        # -------------------------------------------------------------
        # Reward Function Design
        # Balances false alarms, missed threats, and investigation overhead.
        # -------------------------------------------------------------
        if is_benign:
            if action == 0:  # MONITOR
                reward = 5.0
                outcome = "Correct Passive Monitoring"
                explanation = "Benign traffic monitored smoothly. Zero latency, zero wasted analyst triage (+5)."
                is_correct = True
            elif action == 1:  # INVESTIGATE
                reward = -2.0
                outcome = "Unnecessary Investigation"
                explanation = "Wasted analyst triage on harmless benign traffic (-2)."
                is_correct = False
            else:  # BLOCK (action == 2)
                reward = -8.0
                outcome = "False Alarm Block"
                explanation = "Critical mistake: blocked legitimate business/user traffic unnecessarily (-8)."
                is_correct = False

        elif is_suspicious:
            if action == 0:  # MONITOR
                reward = -6.0
                outcome = "Overlooked Suspicious Flow"
                explanation = "Passive monitoring missed probing/suspicious flow (-6)."
                is_correct = False
            elif action == 1:  # INVESTIGATE
                reward = 8.0
                outcome = "Accurate Threat Investigation"
                explanation = "Appropriately triaged suspicious flow; confirmed perimeter integrity (+8)."
                is_correct = True
            else:  # BLOCK (action == 2)
                reward = 2.0
                outcome = "Preemptive Block"
                explanation = "Preemptively blocked suspicious flow (+2). Stopped probe, but triage preferred."
                is_correct = True

        else:  # is_severe attack
            if action == 0:  # MONITOR
                reward = -12.0
                outcome = "Critical Threat Ignored"
                explanation = "Disastrous oversight: allowed active exploit/DDoS to execute unimpeded (-12)."
                is_correct = False
            elif action == 1:  # INVESTIGATE
                reward = 3.0
                outcome = "Delayed Containment"
                explanation = "Investigated active severe exploit (+3); gathered telemetry but delayed block."
                is_correct = True
            else:  # BLOCK (action == 2)
                reward = 10.0
                outcome = "Threat Neutralized"
                explanation = "Critical threat actively terminated and blocked at the firewall (+10)."
                is_correct = True

        # -------------------------------------------------------------
        # Transition Dynamics
        # -------------------------------------------------------------
        early_terminal = False

        if is_severe and action == 0:
            # Overlooked severe attack may lead to system breach
            if self.rng.random() < 0.70:
                early_terminal = True
                explanation += " Exploit escalated into a critical system breach."

        if action == 2:  # BLOCK
            # Blocking connection forces a fresh new flow connection
            next_traffic = self.rng.choice([0, 1, 2], p=[0.50, 0.40, 0.10])
            next_logins = self.rng.choice([0, 1], p=[0.85, 0.15])
            next_behavior = self.rng.choice([0, 1], p=[0.85, 0.15])
            next_state = self.encode_state(next_traffic, next_logins, next_behavior)
        elif action == 1:  # INVESTIGATE
            # Investigation contains threat; system stabilizes
            next_traffic = self.rng.choice([0, 1, 2], p=[0.40, 0.45, 0.15])
            next_logins = 0 if self.rng.random() < 0.75 else 1
            next_behavior = 0 if self.rng.random() < 0.75 else 1
            next_state = self.encode_state(next_traffic, next_logins, next_behavior)
        else:  # MONITOR
            # Environment evolves organically
            curr_decoded = self.decode_state(state)
            t = curr_decoded["traffic"]
            l = curr_decoded["failed_logins"]
            b = curr_decoded["behavior"]

            # Traffic drift
            t_next = int(np.clip(t + self.rng.choice([-1, 0, 1], p=[0.25, 0.50, 0.25]), 0, 2))
            # Login drift
            if l == 1 and self.rng.random() < 0.60:
                l_next = 1  # Persistence of brute-force
            else:
                l_next = self.rng.choice([0, 1], p=[0.80, 0.20])
            # Behavior drift
            if b == 1 and self.rng.random() < 0.60:
                b_next = 1
            else:
                b_next = self.rng.choice([0, 1], p=[0.80, 0.20])

            next_state = self.encode_state(t_next, l_next, b_next)

        # Check termination condition
        time_limit_reached = (self.current_step >= self.max_steps)
        done = early_terminal or time_limit_reached
        self.terminal = done
        self.current_state = next_state

        info = {
            "step": self.current_step,
            "state": state,
            "action": action,
            "action_name": self.ACTIONS[action],
            "next_state": next_state,
            "reward": float(reward),
            "done": done,
            "outcome": outcome,
            "explanation": explanation,
            "is_correct": is_correct,
            "early_terminal": early_terminal,
            "time_limit_reached": time_limit_reached,
            "state_info": self.decode_state(state),
            "next_state_info": self.decode_state(next_state)
        }

        return next_state, float(reward), done, info
