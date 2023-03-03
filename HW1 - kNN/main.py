import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784", as_frame=False)
data = mnist["data"]
labels = mnist["target"]

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


def kNN(n, query_image, k):
    n_train = train[:n]
    n_labels = train_labels[:n]

    dist = calc_dist(n_train, query_image)
    idx_lowest = np.argpartition(dist, k)
    labels_lowest = n_labels[idx_lowest[:k]].astype(int)
    return np.bincount(labels_lowest).argmax()


def get_euclidean_distance(v1, v2):
    dist = np.linalg.norm(v1 - v2)
    return dist


def calc_dist(train_data, query_image):
    dist = np.zeros(len(train_data))
    for i in range(len(train_data)):
        dist[i] = get_euclidean_distance(train_data[i], query_image)
    return dist


def run_algo(k, n):
    results = np.zeros(1000)
    for i in range(1000):
        results[i] = kNN(n, test[i], k)
    return (np.linalg.norm(results == test_labels.astype(int), 1) / 1000) * 100


def section_b():
    n = 1000
    k = 10
    accuracy = run_algo(k, n)
    print("accuracy = " + str(accuracy) + "%")


def section_c():
    n = 1000
    k = np.arange(1, 101)
    results = np.zeros(100)
    for i in range(100):
        print("i = " + str(i))
        results[i] = run_algo(k[i], n)
    plt.plot(k, results)
    plt.show()


def section_d():
    n = np.arange(100, 5001, 100)
    k = 1
    results = np.zeros(50)
    for i in range(50):
        print("i = " + str(i))
        results[i] = run_algo(k, n[i])
    plt.plot(n, results)
    plt.show()


if __name__ == '__main__':
    section_b()
    section_c()
    section_d()
