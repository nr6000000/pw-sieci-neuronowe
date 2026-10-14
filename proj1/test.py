import unittest
from network import *


class TestMLP(unittest.TestCase):
    def test_gradient_check(self):
        pass

    def test_multiclass(self):
        pass


class TestMLPRegression(unittest.TestCase):
    def test_regression_linear(self):
        X = np.linspace(-2, 3, 200).reshape(-1, 1)
        y = 3 * X - 2
        EPOCHS = 800
        model = MLPNetwork(
            input_size=1,
            layers=[
                Layer(1, Activation.linear, Activation.Gradient.linear)
            ],
            loss_function=Loss.mse,
            loss_gradient_function=Loss.Gradient.mse,
            include_bias=True,
            seed=67,
        )

        loss_before = Loss.mse(y, model.predict(X))
        model.fit(
            X, y,
            epochs=EPOCHS,
            learning_rate=0.05,
            batch_size=32,
            history_saved=True
        )
        loss_after = Loss.mse(y, model.predict(X))

        self.assertLess(loss_after, loss_before)
        self.assertLess(loss_after, 0.01)

        X_test = np.array([[-1.276], [0.333], [0.666], [2.4], [2.626]])

        np.testing.assert_allclose(
            model.predict(X_test),
            3 * X_test - 2,
            atol=0.1
        )

        # Czy historia została zapisana?
        n_steps = EPOCHS * ((len(X) + 16 - 1) // 16)
        self.assertEqual(len(model.loss_history), n_steps)
        self.assertEqual(len(model.weight_history), n_steps)

    def test_regression_quadratic(self):
        # y = x^2
        X = np.linspace(-3, 3, 500).reshape(-1, 1)
        y = X ** 2 - 1
        model = MLPNetwork(
            input_size=1,
            layers=[
                Layer(16, Activation.tanh, Activation.Gradient.tanh),
                Layer(1, Activation.linear, Activation.Gradient.linear),
            ],
            loss_function=Loss.mse,
            loss_gradient_function=Loss.Gradient.mse,
            include_bias=True,
            seed=67,
        )

        initial_loss = Loss.mse(y, model.predict(X))
        model.fit(
            X, y,
            epochs=1500,
            learning_rate=0.02,
            batch_size=32
        )

        X_test = X.copy()
        y_test = X_test ** 2 - 1
        test_mse = Loss.mse(y_test, model.predict(X_test))

        self.assertLess(test_mse, 0.02)
        self.assertLess(
            Loss.mse(y, model.predict(X)),
            initial_loss
        )


    def test_dataset_regression_activation(self):
        pass

    def test_dataset_regression_cube(self):
        pass


class TestMLPClassification(unittest.TestCase):# classify (separate) x1 + x2 > 0 vs otherwise
    def test_positive_sum(self):
        pass

    def test_dataset_classification_simple(self):
        pass

    def test_dataset_classification_three_gauss(self):
        pass


if __name__ == '__main__':
    unittest.main()
