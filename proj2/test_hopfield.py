import unittest
from unittest.mock import patch

import numpy as np
from hopfield import HopfieldNet

X = np.ones((1, 4))
noisy = np.array([1, 1, 1, -1])
links = np.ones((4, 4)) - np.eye(4)


class TestHopfield(unittest.TestCase):
    def check_recall(self, net):
        for sync in (True, False):
            with self.subTest(synchronous=sync):
                net.synchronous = sync
                np.testing.assert_array_equal(net.predict(noisy), X[0])
                self.assertEqual(net.last_status, "fixed_point")

    def test_hebb(self):
        net = HopfieldNet(update_rule="hebb").fit(X)
        np.testing.assert_allclose(net.weights, 0.25 * links)
        self.check_recall(net)

    def test_oja(self):
        with patch("hopfield.np.random.normal", return_value=0.2 * links):
            net = HopfieldNet(update_rule="oja", epochs=1).fit(X)
        # y = 0.6, przyrost wagi to 0.001 * (0.6 - 0.6**2 * 0.2).
        np.testing.assert_allclose(net.weights, 0.200528 * links)
        self.check_recall(net)


if __name__ == "__main__":
    unittest.main()
