"""Independent checks for batch Oja subspace learning and memory retrieval."""

import unittest

import numpy as np
from hopfield import HopfieldNet, async_sweep


class OjaSubspaceTests(unittest.TestCase):
    def network(self, *, eta=0.01, steps=10_000, tol=1e-6, seed=42):
        return HopfieldNet(
            synchronous=True,
            update_rule="oja_subspace",
            max_steps=100,
            learning_rate=eta,
            epochs=10,
            max_fit_steps=steps,
            fit_tol=tol,
            fit_check_every=100,
            random_state=seed,
        )

    def test_step_bound_for_large_single_pattern(self):
        x = np.ones(1250)
        net = self.network(eta=0.001)
        with self.assertRaisesRegex(ValueError, "learning_rate <="):
            net.fit(x[None, :])
        net = self.network(eta=1.0 / x.size)
        net.fit(x[None, :])
        self.assertTrue(net.fit_info["converged"])
        self.assertEqual(net.fit_info["learning_rate"], 1.0 / x.size)

    def assert_projector_weights(self, net, projector):
        expected = projector.copy()
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(net.weights, expected, rtol=1e-5, atol=1e-5)

    def test_single_pattern_projector_and_noisy_negation_recovery_with_jit(self):
        x = np.where(np.arange(25) % 3, -1, 1)
        net = self.network()
        net.fit(x[None, :])
        self.assertTrue(net.fit_info["converged"])
        self.assertLessEqual(net.fit_info["residual"], 1e-6)
        self.assert_projector_weights(net, np.outer(x, x) / x.size)
        for synchronous in (True, False):
            net.synchronous = synchronous
            for target in (x, -x):
                noisy = target.copy()
                noisy[[2, 7, 19]] *= -1
                query_before = noisy.copy()
                result = net.predict(noisy)
                np.testing.assert_array_equal(result, target)
                np.testing.assert_array_equal(noisy, query_before)
                self.assertEqual(net.last_status, "fixed_point")
        self.assertTrue(async_sweep.nopython_signatures)

    def test_orthogonal_patterns_match_known_projector(self):
        X = np.array(
            [[(-1.0) ** ((row & col).bit_count()) for col in range(16)] for row in range(4)]
        )
        net = self.network(eta=0.05)
        net.fit(X)
        self.assertTrue(net.fit_info["converged"])
        self.assert_projector_weights(net, X.T @ X / X.shape[1])
        for original in X:
            np.testing.assert_array_equal(net.predict(original), original)
            self.assertEqual(net.last_status, "fixed_point")

    def test_duplicate_and_antipodal_patterns_use_rank_not_row_count(self):
        x = np.array([1, -1, -1, 1, -1, 1, 1, -1, 1])
        X = np.stack([x, x, -x, x])
        net = self.network()
        net.fit(X)
        self.assertEqual(net.fit_info["training_rank"], 1)
        self.assertEqual(net.fit_info["n_components"], 1)
        self.assertEqual(net.basis_.shape, (9, 1))
        self.assert_projector_weights(net, np.outer(x, x) / x.size)
        for original in (x, -x):
            np.testing.assert_array_equal(net.predict(original), original)

    def test_one_step_budget_reports_nonconvergence_and_final_residual(self):
        X = np.array([[1, 1, -1, -1, 1, -1], [-1, 1, 1, -1, -1, 1]])
        net = self.network(steps=1, tol=1e-12)
        net.fit(X)
        self.assertEqual(net.fit_info["fit_steps"], 1)
        self.assertEqual(net.fit_info["data_passes"], 1)
        self.assertFalse(net.fit_info["converged"])
        self.assertGreater(net.fit_info["residual"], 1e-12)
        residual = np.linalg.norm(X - X @ net.basis_ @ net.basis_.T) / np.linalg.norm(X)
        self.assertAlmostEqual(net.fit_info["residual"], residual, places=12)
        self.assertTrue(np.isfinite(net.weights).all())

    def test_full_rank_rejected_and_failed_refit_clears_previous_state(self):
        net = self.network()
        net.fit(np.array([[1, -1, 1, -1, 1]]))
        self.assertIsNotNone(net.weights)
        self.assertIsNotNone(net.basis_)
        with self.assertRaises(ValueError):
            net.fit(np.array([[1, 1], [1, -1]]))
        self.assertIsNone(net.weights)
        self.assertIsNone(net.basis_)
        with self.assertRaises(RuntimeError):
            net.predict(np.array([1, -1, 1, -1, 1]))

    def test_initialization_seed_is_reproducible_and_recorded(self):
        X = np.array([[1, 1, -1, -1, 1, -1], [-1, 1, 1, -1, -1, 1]])
        first = self.network(steps=1, tol=1e-12, seed=19)
        second = self.network(steps=1, tol=1e-12, seed=19)
        different = self.network(steps=1, tol=1e-12, seed=20)
        for net in (first, second, different):
            net.fit(X)
        np.testing.assert_array_equal(first.weights, second.weights)
        np.testing.assert_array_equal(first.basis_, second.basis_)
        self.assertFalse(np.allclose(first.weights, different.weights, atol=1e-10))
        self.assertEqual(first.fit_info["initialization_seed"], 19)
        self.assertEqual(different.fit_info["initialization_seed"], 20)

    def test_one_batch_step_matches_manual_oja_update_and_qr(self):
        X = np.array([[1.0, 1.0, -1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0]])
        eta, seed = 0.01, 17
        rank = int(np.linalg.matrix_rank(X))
        rng = np.random.default_rng(seed)
        initial, _ = np.linalg.qr(rng.normal(size=(X.shape[1], rank)), mode="reduced")
        covariance_times_basis = X.T @ (X @ initial) / len(X)
        step = initial + eta * (
            covariance_times_basis - initial @ (initial.T @ covariance_times_basis)
        )
        expected_basis, _ = np.linalg.qr(step, mode="reduced")
        expected_projector = expected_basis @ expected_basis.T
        net = self.network(eta=eta, steps=1, tol=1e-12, seed=seed)
        net.fit(X)
        np.testing.assert_allclose(
            net.basis_ @ net.basis_.T, expected_projector, rtol=1e-12, atol=1e-12
        )
        expected_weights = expected_projector.copy()
        np.fill_diagonal(expected_weights, 0)
        np.testing.assert_allclose(net.weights, expected_weights, rtol=1e-12, atol=1e-12)

    def test_fit_and_predict_do_not_mutate_input_arrays(self):
        X = np.array([[1, -1, 1, -1, 1], [-1, 1, -1, 1, -1]], dtype=np.float32)
        before = X.copy()
        X.flags.writeable = False
        net = self.network()
        net.fit(X)
        np.testing.assert_array_equal(X, before)
        noisy = before[0].copy()
        noisy[2] *= -1
        noisy_before = noisy.copy()
        noisy.flags.writeable = False
        net.synchronous = False
        net.predict(noisy)
        np.testing.assert_array_equal(noisy, noisy_before)

    def test_learned_basis_and_recurrent_weights_have_required_structure(self):
        X = np.random.default_rng(93).choice([-1.0, 1.0], size=(3, 9))
        net = self.network(eta=0.05)
        net.fit(X)
        rank = int(np.linalg.matrix_rank(X))
        self.assertEqual(net.basis_.shape, (9, rank))
        self.assertEqual(net.basis_.dtype, np.float64)
        self.assertEqual(net.weights.dtype, np.float64)
        self.assertTrue(np.isfinite(net.basis_).all())
        self.assertTrue(np.isfinite(net.weights).all())
        np.testing.assert_allclose(net.basis_.T @ net.basis_, np.eye(rank), atol=1e-12)
        np.testing.assert_array_equal(net.weights, net.weights.T)
        np.testing.assert_array_equal(np.diag(net.weights), np.zeros(9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
