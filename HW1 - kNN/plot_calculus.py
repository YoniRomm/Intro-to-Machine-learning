import matplotlib.pyplot as plt
import numpy as np


def plot():
    for n in [5, 10, 30, 50, 80]:
        x = np.linspace(0, 1, 100)
        y = n * (x ** (n - 1))
        plt.plot(x, y, label=f"n={n}")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()
