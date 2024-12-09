import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("./results/results_acc_mnist.csv", names=["BER", "Accuracy"])
    df['index'] = range(1, len(df) + 1)
    df.set_index('index', inplace=True)

    print(df.describe())

    # plot
    plt.plot(df["BER"], df["Accuracy"])
    plt.title("Accuracy Degradation")
    plt.xlabel("Bit Error Rate")
    plt.ylabel("Model Accuracy")
    plt.show()
