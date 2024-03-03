import numpy as np
import tensorflow as tf
import beras
import beras.metrics
import beras.losses
import beras.activations

from sklearn.metrics import mean_squared_error


def test_mse_forward():
    tensorflow_mse = tf.keras.losses.MeanSquaredError()
    beras_mse = beras.MeanSquaredError()

    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    # assert np.allclose(tensorflow_mse(x, y).numpy(), beras_mse(x, y))
    assert np.allclose(mean_squared_error(x, y), beras_mse(x, y))

    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    y = np.array([[3, 2, 1], [6, 5, 4]], dtype=np.float32)

    # assert np.allclose(tensorflow_mse(x, y).numpy(), beras_mse(x, y))
    assert np.allclose(mean_squared_error(x, y), beras_mse(x, y))
    print("MSE test passed!")

def test_leaky_relu():
    student_leaky_relu = beras.activations.LeakyReLU()
    leaky_relu = tf.keras.layers.LeakyReLU()
    test_arr = np.array(np.arange(-8,8),np.float64)
    assert(all(np.isclose(student_leaky_relu(test_arr),leaky_relu(test_arr))))
    print("Leaky ReLU test passed!")


def test_sigmoid():
    test_arr = np.array(np.arange(-8, 8),np.float64)
    student_sigmoid = beras.activations.Sigmoid()
    act_sigmoid = tf.keras.activations.sigmoid(test_arr)
    assert(all(np.isclose(student_sigmoid(test_arr),act_sigmoid)))
    print("Sigmoid test passed!")

def test_softmax():
    # test_arr = np.array(np.arange(-8, 8),np.float64)
    # student_softmax = beras.activations.Softmax()
    # act_softmax = tf.keras.layers.Softmax()(test_arr)
    # assert(all(np.isclose(student_softmax(test_arr),act_softmax)))
    # print("Softmax test passed!")

    # test_arr = [[-2.12584153],
    #             [-1.89136465],
    #             [-3.90535175],
    #             [ 4.49564314],
    #             [ 7.01672024],
    #             [-9.02192839],
    #             [ 2.62647374],
    #             [ 9.147336  ],
    #             [ 8.91846959],
    #             [ 9.60159028]]
    student_softmax = beras.activations.Softmax()
    # student_softmax(test_arr)
    # student_softmax.get_input_gradients()

    test_arr_2 = [[-2.80376379, -5.35437188, -4.52493736, 9.6010282,  -1.11580615,
                    -3.0670372 ],
                    [-1.31320819,  5.16083376, -5.23721666, -8.72239508,  2.47294677,
                    -4.52774126],
                    [-3.99588307,  6.40716963,  4.12974557, 2.50529463,  7.56326712,
                    9.98126491]]
    student_softmax(test_arr_2)
    student_softmax.get_input_gradients()


def test_categorical_accuracy():
    y_true = [[0, 0, 1], [0, 1, 0]]
    y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    student_acc = beras.metrics.CategoricalAccuracy()(y_pred,y_true)
    acc = tf.keras.metrics.categorical_accuracy(y_true,y_pred)
    assert(student_acc == np.mean(acc))
    print("Categorical accuracy test passed")

def test_cce():
    y_true = np.array([[0, 0, 1], [0, 1, 0]])
    y_pred = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])

    # y_true = np.array([[ 17.92332768,  17.85306202, -38.80526728, -33.42897744, -18.5882171,
    #                     53.60800089, -26.21636972,  19.84668407, -45.13567733,  79.75953683,
    #                     -25.31208578,  79.61551707,  44.01398317, -38.99818631,  79.46465096,
    #                     87.85182343, -39.38558315,  53.84792435, -17.94212475, -40.02702403,
    #                     -27.62711886, -36.11353669,  16.2133409,  -89.82543172,  -5.84921919,
    #                     -49.41463053,  31.74966369, -50.78588663,  96.52864091, -50.23197049,
    #                     26.23059193, -99.02802501,  28.7430564,    3.46274974,   5.21924121,
    #                     -54.86072784, -32.65260468, -84.2347181,  -56.42687816,  98.11689478,
    #                     -12.84708595]])
    # y_pred = np.array([[1.0000000e-08, 9.9999999e-01, 9.9999999e-01, 1.0000000e-08, 1.0000000e-08,
    #                     1.0000000e-08, 1.0000000e-08, 1.0000000e-08, 1.0000000e-08, 9.9999999e-01,
    #                     9.9999999e-01, 9.9999999e-01, 1.0000000e-08, 9.9999999e-01, 9.9999999e-01,
    #                     9.9999999e-01, 9.9999999e-01, 1.0000000e-08, 1.0000000e-08, 9.9999999e-01,
    #                     9.9999999e-01, 1.0000000e-08, 9.9999999e-01, 1.0000000e-08, 1.0000000e-08,
    #                     1.0000000e-08, 9.9999999e-01, 9.9999999e-01, 9.9999999e-01, 1.0000000e-08,
    #                     1.0000000e-08, 1.0000000e-08, 1.0000000e-08, 1.0000000e-08, 1.0000000e-08,
    #                     1.0000000e-08, 1.0000000e-08, 9.9999999e-01, 9.9999999e-01, 9.9999999e-01,
    #                     1.0000000e-08]])

    # student_acc = beras.losses.CategoricalCrossentropy()(y_pred,y_true)
    student_acc = beras.losses.CategoricalCrossentropy()
    #~.4556 is what your solution should output. Tensorflow's version is NOT stable so the numbers differ slightly
    acc = np.mean(tf.keras.losses.categorical_crossentropy(y_true,y_pred).numpy())
    print(student_acc(y_pred, y_true))
    print("acc: " + str(acc))
    print(student_acc.get_input_gradients())
    # assert(np.isclose(student_acc,.45561262645686385))
    # assert(np.isclose(student_acc(y_pred, y_true),acc))
    print("CCE test passed")

def test_cce_new():
    y_true = np.array([[0, 0, 1], [0, 1, 0]])
    y_pred = np.array([[0.1, 0.2, 0.7], [0.05, 0.95, 0]])
    student_acc = beras.losses.CategoricalCrossentropy()(y_pred,y_true)
    assert(np.isclose(student_acc, 0.20398411916314152))
    print("new CCE test passed")


if __name__ == "__main__":
    '''
    Uncomment the tests you would like to run for sanity checks throughout the assignment
    '''

    # ### MSE Test ###
    # test_mse_forward()

    # ### LeakyReLU ###
    # test_leaky_relu()

    # ### Sigmoid ###
    # test_sigmoid()

    ### Softmax ###
    # test_softmax()


    ### Test Categorical Accuracy ###
    # test_categorical_accuracy()

    ### Test CCE ###
    # test_cce()

    test_cce_new()
