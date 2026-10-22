"""
Unit tests for NetworkSecurityEnv.
Verifies state encoding/decoding, reset, transitions, rewards, and reproducibility.
"""

import unittest
import numpy as np
from CyberSentinel_AI.rl.environment import NetworkSecurityEnv


class TestNetworkSecurityEnv(unittest.TestCase):

    def setUp(self):
        self.env = NetworkSecurityEnv(max_steps=20, seed=42)

    def test_state_encoding_decoding_bijective(self):
        """Verify state encoding and decoding match 1-to-1 across all 12 combinations."""
        for traffic in range(3):
            for logins in range(2):
                for behavior in range(2):
                    state_id = NetworkSecurityEnv.encode_state(traffic, logins, behavior)
                    self.assertTrue(0 <= state_id < 12)
                    decoded = NetworkSecurityEnv.decode_state(state_id)
                    self.assertEqual(decoded["traffic"], traffic)
                    self.assertEqual(decoded["failed_logins"], logins)
                    self.assertEqual(decoded["behavior"], behavior)

    def test_reset_behavior(self):
        """Verify reset re-initializes step count and returns valid state."""
        state = self.env.reset(seed=123)
        self.assertTrue(0 <= state < 12)
        self.assertEqual(self.env.current_step, 0)
        self.assertFalse(self.env.terminal)

    def test_reward_design_differentiation(self):
        """Verify that rewards differentiate correctly based on state threat category."""
        # State 0 is benign (Low traffic, normal logins, normal behavior)
        self.env.current_state = 0
        self.env.terminal = False
        _, r_monitor, _, _ = self.env.step(0)  # MONITOR
        self.assertEqual(r_monitor, 5.0)

        self.env.current_state = 0
        self.env.terminal = False
        _, r_block, _, _ = self.env.step(2)  # BLOCK
        self.assertEqual(r_block, -8.0)

        # State 3 is severe (Low traffic, suspicious logins, anomalous behavior)
        self.env.current_state = 3
        self.env.terminal = False
        _, r_sev_block, _, _ = self.env.step(2)  # BLOCK
        self.assertEqual(r_sev_block, 10.0)

        self.env.current_state = 3
        self.env.terminal = False
        _, r_sev_mon, _, _ = self.env.step(0)  # MONITOR
        self.assertEqual(r_sev_mon, -12.0)

    def test_reproducibility(self):
        """Verify identical random seeds produce identical trajectory sequences."""
        env1 = NetworkSecurityEnv(max_steps=10, seed=999)
        env2 = NetworkSecurityEnv(max_steps=10, seed=999)

        s1 = env1.reset(seed=999)
        s2 = env2.reset(seed=999)
        self.assertEqual(s1, s2)

        for a in [0, 1, 2, 0, 1]:
            n1, r1, d1, _ = env1.step(a)
            n2, r2, d2, _ = env2.step(a)
            self.assertEqual(n1, n2)
            self.assertEqual(r1, r2)
            self.assertEqual(d1, d2)
            if d1:
                break


if __name__ == "__main__":
    unittest.main()
