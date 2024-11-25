import tensorflow as tf
import numpy as np
from datetime import datetime
import csv

from tensorflow.keras.models import load_model

from NEBULA import Injector, LegacyInjector

SAMPLESIZE = 10

if __name__ == "__main__":
    # TODO use different models with more layers
    model = load_model("sampledata/mnist_model.h5")

    injector = Injector(model.layers)
    legacy = LegacyInjector(model.layers)

    with open("results_old.csv", "w+") as file_old:
        csvwriter = csv.writer(file_old, delimiter=',')
        for i in range(SAMPLESIZE):
            time_start = datetime.now()
            legacy.injectError(model)
            time_end = datetime.now()
            time = time_end - time_start
            csvwriter.writerow([time.microseconds])
            print(f"Progress: {i}/{SAMPLESIZE}")

        print("Done with legacy measurements")

    with open("results_new.csv", "w+") as file_new:
        csvwriter = csv.writer(file_new, delimiter=",")
        for i in range(SAMPLESIZE):
            time_start = datetime.now()
            injector.injectError(model)
            time_end = datetime.now()
            time = time_end - time_start
            csvwriter.writerow([time.microseconds])
            print(f"Progress: {i}/{SAMPLESIZE}")

        print("Done with new measurements")
