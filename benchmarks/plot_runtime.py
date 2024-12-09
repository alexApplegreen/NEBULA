import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("results_MNIST_SEI_new.csv", names=["new"])
    df2 = pd.read_csv("results_MNIST_SEI_old.csv", names=["old"])

    df = pd.concat([df, df2], axis=1)
    df['index'] = range(1, len(df) + 1)
    df.set_index('index', inplace=True)

    # data
    avg_runtime_new = np.average(df['new'])
    avg_runtime_old = np.average(df['old'])
    df['new'] = df["new"].apply(lambda x: x / 1000)
    df['old'] = df["old"].apply(lambda x: x / 1000)
    std_new = df['new'].std()
    std_old = df['old'].std()

    std_new = df['new'].std()
    std_old = df['old'].std()

    avg_new = np.average(df['new'])
    avg_old = np.average(df['old'])

    print(f"std old: {std_old}, std new: {std_new}, avg old: {avg_old}, avg new: {avg_new}")

    # plot
    fig, ax = plt.subplots()
    VP = ax.boxplot(
        df,
        widths=0.5,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 0.5},
        boxprops={"facecolor": "C0", "edgecolor": "white","linewidth": 0.5},
        whiskerprops={"color": "C0", "linewidth": 1.5},
        capprops={"color": "C0", "linewidth": 1.5}
    )
    positions = [1, 2]
    labels = ['NEBULA', 'Legacy']
    ax.xaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    ax.set_title("Runtime comparison large model")
    ax.set_xlabel("Implementation")
    ax.set_ylabel('Runtime in ms')
    plt.show()
