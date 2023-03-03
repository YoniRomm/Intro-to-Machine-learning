import random
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def q1_a():
    linear_svm = svm.SVC(C=C, kernel="linear")
    popy_svm_2 = svm.SVC(C=C, kernel="poly", degree=2, coef0=0, gamma="auto")
    poly_svm_3 = svm.SVC(C=C, kernel="poly", degree=3, coef0=0, gamma="auto")

    linear_svm.fit(X, y)
    popy_svm_2.fit(X, y)
    poly_svm_3.fit(X, y)

    plot_results([linear_svm, popy_svm_2, poly_svm_3], ["linear", "poly of degree 2", "poly of degree 3"], X, y)


def q1_b():
    linear_svm = svm.SVC(C=C, kernel="linear", coef0=2)
    popy_svm_2 = svm.SVC(C=C, kernel="poly", degree=2, coef0=2, gamma="auto")
    poly_svm_3 = svm.SVC(C=C, kernel="poly", degree=3, coef0=2, gamma="auto")

    linear_svm.fit(X, y)
    popy_svm_2.fit(X, y)
    poly_svm_3.fit(X, y)

    plot_results([linear_svm, popy_svm_2, poly_svm_3], ["linear", "poly of degree 2", "poly of degree 3"], X, y)


def q1_c():
    copy_y = np.copy(y)
    for i in range(len(copy_y)):
        is_change = random.random() < 0.1
        if is_change and copy_y[i] == -1:
            copy_y[i] = 1

    popy_svm_2 = svm.SVC(C=C, kernel="poly", degree=2, coef0=10, gamma="auto")
    popy_svm_2.fit(X, copy_y)
    plot_results([popy_svm_2], ["poly of degree 2"], X, copy_y)

    for gamma in [10, 20, 30, 50, 100]:
        rbf = svm.SVC(C=C, kernel="rbf", gamma=gamma)
        rbf.fit(X, copy_y)
        plot_results([rbf], [f"rbf with gamma = {gamma}"], X, copy_y)


C_hard = 1000000.0  # SVM regularization parameter
C = 10
n = 100

# Data is labeled by a circle

radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
angles = 2 * math.pi * np.random.random(2 * n)
X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
X2 = (radius * np.sin(angles)).reshape((2 * n, 1))

X = np.concatenate([X1, X2], axis=1)
y = np.concatenate([np.ones((n, 1)), -np.ones((n, 1))], axis=0).reshape([-1])
q1_a()
q1_b()
q1_c()
