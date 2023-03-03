from matplotlib import pyplot as plt

if __name__ == '__main__':
    x = [pow(10, k) for k in range(-5, 5)]
    y = x

    plt.plot(x, y, marker='o')
    plt.xlabel("C")
    plt.ylabel("Average Accuracy across 10 runs")
    plt.xscale('log')
    plt.show()
