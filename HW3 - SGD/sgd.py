#################################
# Your name: Yonatan Romm
#################################
import numpy as np
import numpy.random
import scipy
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
from scipy.special import softmax

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """

    n = len(data[0])
    w = np.zeros(n)

    for t in range(1, T + 1):
        i = np.random.randint(0, len(data))
        eta_t = eta_0 / t
        if np.inner(w, data[i]) * labels[i] < 1:
            w = (1 - eta_t) * w + eta_t * C * labels[i] * data[i]
        else:
            w = (1 - eta_t) * w

    return w


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    n = len(data[0])
    w = np.zeros(n)

    for t in range(1, T + 1):
        i = np.random.randint(0, len(data))
        eta_t = eta_0 / t
        x_i = data[i]
        y_i = labels[i]
        gradient = x_i * y_i * scipy.special.softmax([0, -y_i * np.dot(w, x_i)])[1]  # this is the derivative of l_log at point i
        w = w + eta_t * gradient

    return w


#################################

# Place for additional code

#################################

def get_error(data, labels, w):
    error = 0
    for idx, x in enumerate(data):
        predicted_y = predict_y(x, w)
        if predicted_y != labels[idx]:
            error += 1
    return error / len(data)


def predict_y(x, w):
    y_validation = np.inner(x, w)
    if y_validation >= 0:
        return 1
    else:
        return -1


def get_accuracy_log(x, y, w):
    return np.log(1 + np.exp(-y * np.inner(w * x)))


def get_gradients(x, y, w):
    all_soft_max = softmax(w)
    label = int(y)
    all_soft_max[label] = all_soft_max[label] - 1  # subtruct I{i=y}
    gradients = []
    for i in range(10):
        gradients.append(all_soft_max[i] * x)

    return gradients


def q1a(train_data, train_labels, validation_data, validation_labels):
    eta_array = [pow(10, k) for k in range(-5, 5)]
    accuracy_array = np.zeros(len(eta_array))
    C = 1
    for idx, eta in enumerate(eta_array):
        sum_error = 0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta, 1000)
            sum_error += get_error(validation_data, validation_labels, w)
        accuracy_array[idx] = 1 - (sum_error / 10)

    best_eta = eta_array[np.argmax(accuracy_array)]
    print(f"best eta is {best_eta}")

    plt.plot(eta_array, accuracy_array, marker='o')
    plt.title("q1a")
    plt.xlabel("Eta")
    plt.ylabel("Average Accuracy across 10 runs")
    plt.xscale('log')
    plt.show()


def q1b(train_data, train_labels, validation_data, validation_labels):
    eta = 1  # found in q1a
    C_array = [pow(10, k) for k in range(-5, 5)]
    accuracy_array = np.zeros(len(C_array))
    for idx, C in enumerate(C_array):
        sum_error = 0
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta, 1000)
            sum_error += get_error(validation_data, validation_labels, w)
        accuracy_array[idx] = 1 - (sum_error / 10)

    best_C = C_array[np.argmax(accuracy_array)]
    print(f"best C is {best_C}")

    plt.plot(C_array, accuracy_array, marker='o')
    plt.title("q1b")
    plt.xlabel("C")
    plt.ylabel("Average Accuracy across 10 runs")
    plt.xscale('log')
    plt.show()


def q1c(train_data, train_labels):
    eta = 1  # found in q1a
    C = pow(10, -4)  # found in q1b
    T = 2000  # specified in the question
    w = SGD_hinge(train_data, train_labels, C, eta, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()


def q1d(train_data, train_labels, test_data, test_labels):
    eta = 1  # found in q1a
    C = pow(10, -4)  # found in q1b
    T = 2000  # specified in the question
    w = SGD_hinge(train_data, train_labels, C, eta, T)
    accuracy = 1 - get_error(test_data, test_labels, w)
    print(f'best accuracy q1d {accuracy}')


def q2a(train_data, train_labels, validation_data, validation_labels):
    eta_array = [pow(10, k) for k in range(-10, 5)]
    accuracy_array = np.zeros(len(eta_array))
    for idx, eta in enumerate(eta_array):
        sum_error = 0
        for i in range(10):
            w = SGD_log(train_data, train_labels, eta, 1000)
            sum_error += get_error(validation_data, validation_labels, w)
        accuracy_array[idx] = 1 - sum_error / 10

    best_eta = eta_array[np.argmax(accuracy_array)]
    print(f"best eta is {best_eta}")

    plt.plot(eta_array, accuracy_array, marker='o')
    plt.title("q2a")
    plt.xlabel("Eta")
    plt.ylabel("Average Accuracy across 10 runs")
    plt.xscale('log')
    plt.show()


def q2b(train_data, train_labels, test_data, test_labels):
    eta = pow(10, -6)  # found in q2a
    T = 2000  # specified in the question
    w = SGD_log(train_data, train_labels, eta, T)
    accuracy = 1 - get_error(test_data, test_labels, w)
    print(f'best accuracy q2b {accuracy}')
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()


def q2c(train_data, train_labels):
    eta = pow(10, -6)  # found in q2a
    T = 30000  # specified in the question

    w = np.zeros(len(train_data[0]))
    iter_array = np.arange(0, T + 1)
    w_norm = np.zeros(T + 1)

    for t in range(1, T + 1):
        i = np.random.randint(0, len(train_data))
        x_i = train_data[i]
        y_i = train_labels[i]
        eta_t = eta / t
        gradient = x_i * y_i * scipy.special.softmax([0, -y_i * np.dot(w, x_i)])[1]  # this is the derivative of l_log at point i
        w = w + eta_t * gradient
        w_norm[t] = np.linalg.norm(w)

    plt.plot(iter_array, w_norm)
    plt.title("q2c")
    plt.xlabel("Iteration")
    plt.ylabel("W norm")
    plt.show()


if __name__ == '__main__':
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    q1a(train_data, train_labels, validation_data, validation_labels)
    q1b(train_data, train_labels, validation_data, validation_labels)
    q1c(train_data, train_labels)
    q1d(train_data, train_labels, test_data, test_labels)
    q2a(train_data, train_labels, validation_data, validation_labels)
    q2b(train_data, train_labels, test_data, test_labels)
    q2c(train_data, train_labels)
