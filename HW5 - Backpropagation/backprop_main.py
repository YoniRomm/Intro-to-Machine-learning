import numpy as np
from matplotlib import pyplot as plt

import backprop_data

import backprop_network


def one_a():
    training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)

    net = backprop_network.Network([784, 40, 10])

    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)


def one_b():
    training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)

    rates = [0.001, 0.01, 0.1, 1, 10, 100]

    test_accuracies = [0] * len(rates)
    training_accuracies = [0] * len(rates)
    training_losses = [0] * len(rates)

    for i in range(len(rates)):
        net = backprop_network.Network([784, 40, 10])
        test_accuracies[i], training_accuracies[i], training_losses[i] = net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=rates[i],
                                                                                 test_data=test_data)

    # PLOT 1

    for i in range(len(rates)):
        plt.plot(np.arange(30), test_accuracies[i], label=f"rate = {rates[i]}")

    plt.title('Test Accuracy by Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.show()

    # PLOT 2

    for i in range(len(rates)):
        plt.plot(np.arange(30), training_losses[i], label=f"rate = {rates[i]}")

    plt.title('Training Loss by Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.show()

    # PLOT 3

    for i in range(len(rates)):
        plt.plot(np.arange(30), training_accuracies[i], label=f"rate = {rates[i]}")

    plt.title('Training Accuracy by Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.legend()
    plt.show()


def one_c():
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)


def one_d():
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
    network_array = [784, 500, 10]
    net = backprop_network.Network(network_array)
    epochs = 30
    batch = 2
    rate = 0.01

    print(f'running with net: {network_array}, epochs: {epochs}, batch: {batch}, rate: {rate}')
    net.SGD(training_data, epochs=epochs, mini_batch_size=batch, learning_rate=rate, test_data=test_data)
