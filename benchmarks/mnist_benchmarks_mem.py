from datetime import datetime
import csv
import tracemalloc

from tensorflow.keras.models import load_model

from NEBULA.core import Injector
from NEBULA.core.legacyInjector import LegacyInjector

SAMPLESIZE = 200  # Modify this to set number of measurements
COUNTER = 0

def avgtime(time, i):
    time_sum = COUNTER
    time_sum += time.seconds
    time_left = 0
    if i != 0:
        time_avg = time_sum / i
        time_left = time_avg * (SAMPLESIZE - i)
    return time_left

if __name__ == "__main__":
    # TODO use different models with more layers
    model = load_model("../sample/sampledata/big_mnist_model.h5")

    tracemalloc.start()
    with open("results_MNIST_SEI_mem_old.csv", "w+") as file_old:
        csvwriter = csv.writer(file_old, delimiter=",")
        for i in range(SAMPLESIZE):
            time_start = datetime.now()
            injector = LegacyInjector(model.layers)

            tracemalloc.reset_peak()

            injector.injectError(model)
            _, peak_mem = tracemalloc.get_traced_memory()

            tracemalloc.reset_peak()
            time_end = datetime.now()
            time = time_end - time_start
            csvwriter.writerow([peak_mem])
            COUNTER += time.seconds
            time_left = avgtime(time, i)
            print(f"Progress: {i}/{SAMPLESIZE}, projected time left: {time_left}s")
        tracemalloc.stop()

    tracemalloc.start()
    with open("results_MNIST_SEI_mem_new.csv", "w+") as file_new:
        csvwriter = csv.writer(file_new, delimiter=",")
        for i in range(SAMPLESIZE):
            time_start = datetime.now()
            injector = Injector(model.layers)

            tracemalloc.reset_peak()

            injector.injectError(model)
            _, peak_mem = tracemalloc.get_traced_memory()

            tracemalloc.reset_peak()
            time_end = datetime.now()
            time = time_end - time_start
            csvwriter.writerow([peak_mem])
            COUNTER += time.seconds
            time_left = avgtime(time, i)
            print(f"Progress: {i}/{SAMPLESIZE}, projected time left: {time_left}s")
        tracemalloc.stop()

        print("Done with new measurements")
