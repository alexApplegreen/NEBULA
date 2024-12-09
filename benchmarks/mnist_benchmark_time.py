from datetime import datetime
import csv

from tensorflow.keras.models import load_model

from NEBULA.core import Injector
from NEBULA.core.legacyInjector import LegacyInjector

SAMPLESIZE = 5  # Modify this to set number of measurements
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
    model = load_model("../sample/sampledata/big_mnist_model.h5")

    with open("results_MNIST_SEI_old.csv", "w+") as file_old:
        csvwriter = csv.writer(file_old, delimiter=",")
        injector = LegacyInjector(model.layers)

        for i in range(SAMPLESIZE):

            time_start = datetime.now()
            injector.injectError(model)
            time_end = datetime.now()

            time = time_end - time_start
            csvwriter.writerow([time.microseconds])
            COUNTER += time.seconds
            time_left = avgtime(time, i)

            print(f"Progress: {i}/{SAMPLESIZE}, projected time left: {time_left}s")

    with open("results_MNIST_SEI_new.csv", "w+") as file_new:
        csvwriter = csv.writer(file_new, delimiter=",")
        injector = Injector(model.layers)
        for i in range(SAMPLESIZE):

            time_start = datetime.now()
            injector.injectError(model)
            time_end = datetime.now()

            time = time_end - time_start
            csvwriter.writerow([time.microseconds])
            COUNTER += time.seconds
            time_left = avgtime(time, i)
            print(f"Progress: {i}/{SAMPLESIZE}, projected time left: {time_left}s")

        print("Done with new measurements")
