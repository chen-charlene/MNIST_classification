import numpy as np
from beras.onehot import OneHotEncoder
from beras.core import Tensor
import tensorflow as tf
# from tensorflow.keras import datasets


def load_and_preprocess_data() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    '''This is where we load in and preprocess our data! We load in the data 
        for you but you'll need to flatten the images, normalize the values and 
        convert the input images from numpy arrays into tensors
    Return the preprocessed training and testing data and labels!'''
    
    #Load in the training and testing data from the MNIST dataset
    (train_inputs, train_labels), (test_inputs, test_labels) = tf.keras.datasets.mnist.load_data()
    ## TODO: Flatten (reshape) and normalize the inputs
    ## Hint: train and test inputs are numpy arrays so you can use np methods on them!

    # flatten inputs into one vector
    flat_train_inputs = np.reshape(train_inputs, [len(train_inputs), 784])
    flat_test_inputs = np.reshape(test_inputs, [len(test_inputs), 784])
    # normalize inputs
    norm_train_inputs = (flat_train_inputs / 255.).astype(np.float32)
    norm_test_inputs = (flat_test_inputs / 255.).astype(np.float32)

    ...
    ## TODO: Convert all of the data into Tensors, the constructor is already
    ##       written for you in Beras/core.py and we import it in line 3
    train_inputs = Tensor(norm_train_inputs)
    train_labels = Tensor(train_labels)
    test_inputs = Tensor(norm_test_inputs)
    test_labels = Tensor(test_labels)
    ...
    return train_inputs, train_labels, test_inputs, test_labels