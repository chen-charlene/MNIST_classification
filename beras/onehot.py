import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    TODO:
        - fit
        - call
        - inverse

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        ## TODO: Fetch all the unique labels and create a dictionary with
        ## the unique labels as keys and their one hot encodings as values
        
        set_data = np.unique(data)
        self.one_hot_dict = {key: i for i, key in enumerate(set_data)}
        self.code_to_label_dict = {}
        
        for key in self.one_hot_dict:
            # one_hot_vector = [0]*len(set_data)
            one_hot_vector = np.zeros(len(set_data))
            index = self.one_hot_dict[key]
            one_hot_vector[index] = 1
            self.one_hot_dict[key] = one_hot_vector
            self.code_to_label_dict[tuple(one_hot_vector)] = key

        

    def call(self, data):
        ## TODO: Implement call function
        output = []
        for i in range(len(data)):
            output.append(self.one_hot_dict[data[i]])
        return np.array(output)


    def inverse(self, data):
        ## TODO: Implement inverse function
        output = []
        for row in data:
            output.append(self.code_to_label_dict[tuple(row)])
        return output