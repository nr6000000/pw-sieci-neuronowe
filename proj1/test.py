import unittest
from network import *

class TestMLPRegression(unittest.TestCase):
    def setUp(self):
        self.mlp = MLPNetwork()

    # regression towards y = 3x - 2
    def test_regression_linear(self):
        pass

    def test_dataset_regression_activation(self):
        pass

    def test_dataset_regression_cube(self):
        pass


class TestMLPClassification(unittest.TestCase):
    def setUp(self):
        self.mlp = MLPNetwork()

    # classify (separate) x1 + x2 > 0 vs otherwise
    def test_positive_sum(self):
        pass

    def test_dataset_classification_simple(self):
        pass

    def test_dataset_classification_three_gauss(self):
        pass


if __name__ == '__main__':
    unittest.main()
