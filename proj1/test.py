import unittest
from pathlib import Path

import numpy as np
from activations import Activation
from losses import Loss
from network import Layer, MLPNetwork

DATA = Path(__file__).resolve().parent / "dataset"


def load_csv(task, name, split):
    path = DATA / task / f"data.{name}.{split}.100.csv"
    return np.loadtxt(path, delimiter=",", skiprows=1)


class TestMLP(unittest.TestCase):
    def _gradient_check(self, model, X, y, eps=1e-5):
        model._forward(X, cached_activations=True)
        grad_w, grad_b = model._backward(y)

        for params, grads in (
            (model.weights, grad_w),
            (model.biases, grad_b),
        ):
            for param, grad in zip(params, grads):
                for idx in np.ndindex(param.shape):
                    original = param[idx]
                    try:
                        param[idx] = original + eps
                        plus = model.loss_fn(y, model.predict(X))

                        param[idx] = original - eps
                        minus = model.loss_fn(y, model.predict(X))
                    finally:
                        param[idx] = original

                    numerical = (plus - minus) / (2 * eps)

                    np.testing.assert_allclose(grad[idx], numerical, rtol=1e-4, atol=1e-6)

    def test_gradient_check_mse(self):
        X = np.array([[0.2, 0.5], [0.7, 0.3], [0.1, 0.9]])
        y = np.array([[0.4], [0.8], [0.2]])

        model = MLPNetwork(
            2,
            [
                Layer(3, Activation.tanh, Activation.Gradient.tanh),
                Layer(1, Activation.linear, Activation.Gradient.linear),
            ],
            Loss.mse,
            Loss.Gradient.mse,
            True,
            42,
        )

        self._gradient_check(model, X, y)

    def test_gradient_check_cce(self):
        X = np.array([[0.2, 0.5], [0.7, 0.3], [0.1, 0.9]])
        y = np.eye(3)[[0, 2, 1]]

        model = MLPNetwork(
            2,
            [
                Layer(4, Activation.tanh, Activation.Gradient.tanh),
                Layer(3, Activation.softmax, Activation.Gradient.softmax),
            ],
            Loss.cce,
            Loss.Gradient.cce,
            True,
            42,
        )

        self._gradient_check(model, X, y)

    def test_multiclass(self):
        pass


class TestMLPRegression(unittest.TestCase):
    def test_regression_linear(self):
        X = np.linspace(-2, 3, 200).reshape(-1, 1)
        y = 3 * X - 2
        EPOCHS = 800
        BATCH_SIZE = 32
        model = MLPNetwork(
            input_size=1,
            layers=[Layer(1, Activation.linear, Activation.Gradient.linear)],
            loss_function=Loss.mse,
            loss_gradient_function=Loss.Gradient.mse,
            include_bias=True,
            seed=67,
        )

        loss_before = Loss.mse(y, model.predict(X))
        model.fit(
            X, y, epochs=EPOCHS, learning_rate=0.05, batch_size=BATCH_SIZE, history_saved=True
        )
        loss_after = Loss.mse(y, model.predict(X))

        self.assertLess(loss_after, loss_before)
        self.assertLess(loss_after, 0.01)

        X_test = np.array([[-1.276], [0.333], [0.666], [2.4], [2.626]])

        np.testing.assert_allclose(model.predict(X_test), 3 * X_test - 2, atol=0.1)

        n_steps = EPOCHS * ((len(X) + BATCH_SIZE - 1) // BATCH_SIZE)
        self.assertEqual(len(model.loss_history), n_steps)
        self.assertEqual(len(model.weight_history), n_steps)

    def test_regression_quadratic(self):
        rng = np.random.default_rng(123)

        X = rng.uniform(-2, 2, (200, 1))
        y = X**2 - 1
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
        model.fit(X, y, epochs=1500, learning_rate=0.02, batch_size=32)

        X_test = np.random.default_rng(456).uniform(-2, 2, (100, 1))
        y_test = X_test**2 - 1
        test_mse = Loss.mse(y_test, model.predict(X_test))

        self.assertLess(test_mse, 0.02)
        self.assertLess(Loss.mse(y, model.predict(X)), initial_loss)

    def _check_dataset_regression(self, name):
        train = load_csv("regression", name, "train")
        test = load_csv("regression", name, "test")

        X, y = train[:, :1], train[:, 1:2]
        X_test, y_test = test[:, :1], test[:, 1:2]

        # norm
        xm, xs = X.mean(axis=0), X.std(axis=0)
        ym, ys = y.mean(axis=0), y.std(axis=0)
        Xn = (X - xm) / xs
        yn = (y - ym) / ys

        model = MLPNetwork(
            1,
            [
                Layer(20, Activation.tanh, Activation.Gradient.tanh),
                Layer(20, Activation.tanh, Activation.Gradient.tanh),
                Layer(1, Activation.linear, Activation.Gradient.linear),
            ],
            Loss.mse,
            Loss.Gradient.mse,
            True,
            67,
        )
        initial_loss = Loss.mse(yn, model.predict(Xn))
        model.fit(Xn, yn, epochs=900, learning_rate=0.02, batch_size=32)
        self.assertLess(Loss.mse(yn, model.predict(Xn)), initial_loss)

        mask = (X_test[:, 0] >= X.min()) & (X_test[:, 0] <= X.max())
        prediction = model.predict((X_test[mask] - xm) / xs) * ys + ym
        baseline = np.full_like(y_test[mask], ym.item())

        self.assertTrue(np.isfinite(prediction).all())
        self.assertLess(Loss.mse(y_test[mask], prediction), Loss.mse(y_test[mask], baseline))

    def test_dataset_regression_activation(self):
        self._check_dataset_regression("activation")

    def test_dataset_regression_cube(self):
        self._check_dataset_regression("cube")


class TestMLPClassification(unittest.TestCase):
    def _check_dataset_classification(self, name, classes):
        train = load_csv("classification", name, "train")
        test = load_csv("classification", name, "test")

        X = train[:, :2]
        X_test = test[:, :2]

        # labels: 1, 2, 3
        labels = train[:, 2].astype(int) - 1
        y_test = test[:, 2].astype(int) - 1

        if classes == 2:
            y = labels.reshape(-1, 1).astype(float)
            output = Layer(1, Activation.sigmoid, Activation.Gradient.sigmoid)
            loss, grad = Loss.bce, Loss.Gradient.bce
        else:
            y = np.eye(classes)[labels]
            output = Layer(classes, Activation.softmax)
            loss, grad = Loss.cce, Loss.Gradient.cce

        model = MLPNetwork(
            2,
            [
                Layer(12, Activation.tanh, Activation.Gradient.tanh),
                output,
            ],
            loss,
            grad,
            True,
            42,
        )

        initial_loss = loss(y, model.predict(X))
        model.fit(X, y, epochs=500, learning_rate=0.05, batch_size=32)
        self.assertLess(loss(y, model.predict(X)), initial_loss)

        probabilities = model.predict(X_test)
        if classes == 2:
            predicted = (probabilities[:, 0] >= 0.5).astype(int)
        else:
            predicted = np.argmax(probabilities, axis=1)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)

        accuracy = np.mean(predicted == y_test)
        self.assertGreater(accuracy, 0.55)

    def test_positive_sum(self):
        rng = np.random.default_rng(123)

        X = rng.uniform(-2, 2, (400, 2))
        y = (X.sum(axis=1) > 0).astype(float).reshape(-1, 1)

        model = MLPNetwork(
            2,
            [
                Layer(6, Activation.tanh, Activation.Gradient.tanh),
                Layer(1, Activation.sigmoid, Activation.Gradient.sigmoid),
            ],
            Loss.bce,
            Loss.Gradient.bce,
            True,
            42,
        )

        model.fit(X, y, epochs=350, learning_rate=0.05, batch_size=32)
        X_test = np.random.default_rng(999).uniform(-2, 2, (150, 2))
        y_test = (X_test.sum(axis=1) > 0).astype(int)

        predicted = (model.predict(X_test).ravel() >= 0.5).astype(int)
        accuracy = np.mean(predicted == y_test)
        self.assertGreaterEqual(accuracy, 0.95)

    def test_dataset_classification_simple(self):
        self._check_dataset_classification("simple", 2)

    def test_dataset_classification_three_gauss(self):
        self._check_dataset_classification("three_gauss", 3)


if __name__ == "__main__":
    unittest.main()
