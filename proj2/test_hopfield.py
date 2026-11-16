import unittest

import numpy as np
from hopfield import HopfieldNet


class HopfieldTests(unittest.TestCase):
    def network(self, rule="oja", synchronous=True, eta=0.001, epochs=10, limit=100):
        return HopfieldNet(synchronous, rule, limit, eta, epochs)

    def test_single_pattern_closed_form_and_seed_independence(self):
        x = np.where(np.arange(25) % 2, -1.0, 1.0)
        expected = (1 - (1 - 0.2) ** 7) * (np.outer(x, x) - np.eye(x.size))
        for seed in range(5):
            np.random.seed(seed)
            net = self.network(eta=0.2, epochs=7)
            net.fit(x[None, :])
            np.testing.assert_allclose(net.weights, expected, atol=1e-14, rtol=1e-13)
            np.testing.assert_array_equal(net.predict(x), x)
            np.testing.assert_array_equal(net.predict(-x), -x)

    def test_multiple_patterns_match_weighted_sum(self):
        X = np.array([[1, 1, -1, -1], [1, -1, 1, -1]], dtype=float)
        eta, epochs, seed = 0.1, 3, 31
        np.random.seed(seed)
        sequence = [X[k] for _ in range(epochs) for k in np.random.permutation(len(X))]
        expected = sum(
            eta * (1 - eta) ** (len(sequence) - 1 - i) * np.outer(x, x)
            for i, x in enumerate(sequence)
        )
        np.fill_diagonal(expected, 0)
        np.random.seed(seed)
        net = self.network(eta=eta, epochs=epochs)
        net.fit(X)
        np.testing.assert_allclose(net.weights, expected, atol=1e-14, rtol=1e-13)

    def test_weight_bounds_symmetry_and_zero_diagonal(self):
        X = np.random.default_rng(8).choice([-1.0, 1.0], size=(5, 25))
        for eta in (0.001, 0.2, 1.0):
            net = self.network(eta=eta)
            net.fit(X)
            self.assertTrue(np.isfinite(net.weights).all())
            self.assertLessEqual(np.max(np.abs(net.weights)), 1.0)
            np.testing.assert_array_equal(net.weights, net.weights.T)
            np.testing.assert_array_equal(np.diag(net.weights), 0)

    def test_orthogonal_patterns_remain_fixed_points(self):
        X = np.array(
            [[(-1.0) ** ((row & col).bit_count()) for col in range(16)] for row in range(4)]
        )
        net = self.network()
        net.fit(X)
        for x in X:
            np.testing.assert_array_equal(net.predict(x), x)
            self.assertEqual(net.last_status, "fixed_point")

    def test_real_jit_recovers_corrupted_integer_input(self):
        x = np.where(np.arange(25) % 2, -1, 1)
        net = self.network(synchronous=False)
        net.fit(x[None, :])
        noisy = x.copy()
        noisy[[0, 4, 7]] *= -1
        for dtype in (np.int64, np.float32, np.float64):
            query = noisy.astype(dtype)
            np.testing.assert_array_equal(net.predict(query), x)
            self.assertEqual(net.last_status, "fixed_point")
            np.testing.assert_array_equal(query, noisy)

    def test_hebb_formula_unchanged(self):
        X = np.array([[1, 1, -1], [1, -1, 1]], dtype=float)
        expected = X.T @ X / len(X)
        np.fill_diagonal(expected, 0)
        net = self.network(rule="hebb")
        net.fit(X)
        np.testing.assert_array_equal(net.weights, expected)

    def test_history_limits_and_two_cycle(self):
        x = np.array([1, -1])
        for limit, expected, status in (
            (0, [[1, -1]], "max_steps"),
            (1, [[1, -1], [-1, 1]], "max_steps"),
            (100, [[1, -1], [-1, 1], [1, -1]], "cycle_2"),
        ):
            net = self.network(rule="hebb", limit=limit)
            net.fit(np.array([[1, 1]]))
            history = net.predict(x, save_steps=True)
            self.assertEqual([state.tolist() for state in history], expected)
            self.assertEqual(net.last_status, status)
            self.assertEqual(net.last_steps, len(history) - 1)
            np.testing.assert_array_equal(history[-1], net.predict(x))
            np.testing.assert_array_equal(x, [1, -1])
            for i, state in enumerate(history):
                self.assertFalse(np.shares_memory(state, x))
                for earlier in history[:i]:
                    self.assertFalse(np.shares_memory(state, earlier))

    def test_invalid_training_clears_old_weights(self):
        net = self.network()
        for eta in (0, -1, 1.01, np.nan, np.inf):
            net.learning_rate = 0.001
            net.fit(np.array([[1, -1]]))
            net.learning_rate = eta
            with self.assertRaises(ValueError):
                net.fit(np.array([[1, -1]]))
            self.assertIsNone(net.weights)
            with self.assertRaises(RuntimeError):
                net.predict(np.array([1, -1]))
        for X in (np.empty((0, 2)), np.array([1, -1]), np.array([[1, np.nan]])):
            with self.assertRaises(ValueError):
                self.network().fit(X)
        for epochs in (0, -1, 0.5):
            with self.assertRaises(ValueError):
                self.network(epochs=epochs).fit(np.array([[1, -1]]))

    def test_nonfinite_weights_rejected(self):
        net = self.network()
        net.fit(np.array([[1, -1]]))
        net.weights[0, 1] = np.nan
        with self.assertRaises(FloatingPointError):
            net.predict(np.array([1, -1]))


if __name__ == "__main__":
    unittest.main()
