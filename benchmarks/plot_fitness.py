import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

if __name__ == "__main__":
    df = pd.read_csv("./results/results_acc_mnist.csv", names=["BER", "Accuracy"])
    df['index'] = range(1, len(df) + 1)
    df.set_index('index', inplace=True)

    print(df.describe())

    # smooth
    window_size = 300  # Choose an appropriate window size
    y = moving_average(df["Accuracy"], window_size)
    x = df["BER"][:len(y)]

    # plot
    plt.plot(x, y)
    plt.title("Accuracy Degradation")
    plt.xlabel("Bit Error Rate")
    plt.ylabel("Model Accuracy")
    plt.show()
