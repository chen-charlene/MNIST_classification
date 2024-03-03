import numpy as np

from beras.core import Callable


class CategoricalAccuracy(Callable):
    """
    TODO:
        - call
    """
    def call(self, probs, labels):
        ## TODO: Compute and return the categorical accuracy of your model 
        ## given the output probabilities and true labels. 
        ## HINT: Argmax + boolean mask via '=='

        probs_array = np.array(probs)
        labels_array = np.array(labels)
        probs_max = np.argmax(probs_array,axis=1)
        labels_max = np.argmax(labels_array,axis=1)
        accuracy = (probs_max == labels_max).mean()

        return accuracy
