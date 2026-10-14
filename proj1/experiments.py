from collections.abc import Callable


class Experiments:
    def __init__(self, test_function: Callable):
        self.test_function = test_function

    def varying_activation(self):
        pass

    def varying_hidden_layers(self):
        pass

    def varying_regression_loss(self):
        pass

    def varying_classification_loss(self):
        pass
