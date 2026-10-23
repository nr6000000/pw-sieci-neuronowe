from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from activations import Activation
from losses import Loss
from network import Layer, MLPNetwork
from plotter import Plotter

DATA = Path(__file__).resolve().parent / "dataset"

def run_regression():
    train = np.loadtxt(DATA / "regression" / "data.activation.train.1000.csv", delimiter=",", skiprows=1)
    test = np.loadtxt(DATA / "regression" / "data.activation.test.1000.csv", delimiter=",", skiprows=1)

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
            Layer(5, Activation.relu, Activation.Gradient.relu),
            Layer(5, Activation.relu, Activation.Gradient.relu),
            Layer(1, Activation.linear, Activation.Gradient.linear),
        ],
        Loss.mse,
        Loss.Gradient.mse,
        True,
        67,
    )
    initial_loss = Loss.mse(yn, model.predict(Xn))
    model.fit(Xn, yn, epochs=900, learning_rate=0.02, batch_size=32, history_saved=True)
    mask = (X_test[:, 0] >= X.min()) & (X_test[:, 0] <= X.max())
    prediction = model.predict((X_test - xm) / xs) * ys + ym
    return model, X_test, y_test, prediction

def run_classification():
    train = np.loadtxt(DATA / "classification" / "data.three_gauss.train.1000.csv", delimiter=",", skiprows=1)
    test = np.loadtxt(DATA / "classification" / "data.three_gauss.test.1000.csv", delimiter=",", skiprows=1)

    X = train[:, :2]
    X_test = test[:, :2]

    # labels: 1, 2, 3
    labels = train[:, 2].astype(int) - 1
    y_test = test[:, 2].astype(int) - 1

    y = np.eye(3)[labels]
    output = Layer(3, Activation.softmax)
    loss, grad = Loss.cce, Loss.Gradient.cce

    model = MLPNetwork(
        2,
        [
            Layer(5, Activation.relu, Activation.Gradient.tanh),
            Layer(5, Activation.relu, Activation.Gradient.tanh),
            output,
        ],
        loss,
        grad,
        True,
        42,
    )

    initial_loss = loss(y, model.predict(X))
    model.fit(X, y, epochs=500, learning_rate=0.05, batch_size=32)

    probabilities = model.predict(X_test)
    predicted = np.argmax(probabilities, axis=1)
    return model, X_test, y_test, predicted

if __name__ == "__main__":
    model, X_test, y_test, y_pred = run_regression()

    Plotter.plot_losses(model.loss_history)
    plt.show(block=True)

    Plotter.plot_weights(model.weight_history, model.bias_history, 
                         interval=1000, num_frames=10)
    plt.show(block=True)

    Plotter.plot_regression_results(X_test, y_test, 'activation', 'real')
    Plotter.plot_regression_results(X_test, y_pred, 'activation', 'pred')
    plt.show(block=True)

    Plotter.plot_lineweight(model.weight_history, (1, 1, 1))
    Plotter.plot_linebias(model.bias_history, (1, 1))
    Plotter.plot_lineweight(model.weight_history, (0, 0, 0))
    Plotter.plot_linebias(model.bias_history, (0, 4))
    plt.show(block=True)

    model, X_test, y_test, y_pred = run_classification()

    Plotter.plot_classification_results(X_test, y_pred, 'three_gauss')
    plt.show(block=True)

    Plotter.plot_classification_results(X_test, y_test, 'three_gauss')
    plt.show(block=True)

