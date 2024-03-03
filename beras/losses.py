import numpy as np

from beras.core import Diffable, Tensor


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    """
    TODO:
        - call function
        - input_gradients
    Identical to HW1!
    """

    def call(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        y_pred: the predicted labels
        y_true: the true labels
        returns: the MeanSquaredError as a Tensor
        """
        differences = y_true - y_pred
        differences_sq = differences * differences
        MeanSquaredError = np.mean(differences_sq, axis=-1)
        return Tensor(np.mean(MeanSquaredError,axis=0))
        

    def get_input_gradients(self) -> list[Tensor]:
        """
        i.e. return the gradient of the layer w.r.t y_pred, the gradient of the layer w.r.t. y_true

        returns: a list of input gradients in the same order as the input arguments of the call function.
        HINT: What would the gradients be with respect to a scalar?
        """
        y_pred = self.inputs[0]
        y_true = self.inputs[1]
        differences = y_true - y_pred
        grad_y_pred = -2 * differences / (len(y_pred) * len(y_pred[0]))
        grad_y_true = 0 * differences / (len(y_true) * len(y_true[0]))
        return [grad_y_pred, grad_y_true]


def clip_0_1(x, eps=1e-8):
    return np.clip(x, eps, 1 - eps)


class CategoricalCrossentropy(Loss):
    """
    TODO: Implement CategoricalCrossentropy class
        - call function
        - input_gradients

        Hint: Use clip_0_1 to stabilize calculations
    """

    def call(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""

        y_pred_norm = clip_0_1(y_pred)
        loss = -np.sum((y_true * np.log(y_pred_norm)))/len(y_pred)
        return loss

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""

        n = len(self.inputs[0])
        y_pred = clip_0_1(self.inputs[0])
        y_true = self.inputs[1]
        grad_y_pred = -(y_true/y_pred)/len(y_pred)
        grad_y_true = -1/n * np.log(y_pred)
        return [grad_y_pred, grad_y_true]
