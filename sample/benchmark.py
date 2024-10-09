import keras
import time

from NEBULA import LegacyInjector, Injector
from NEBULA.core import legacyInjector

if __name__ == '__main__':
    inputs = keras.Input(shape=(37,))
    x = keras.layers.Dense(32, activation="relu")(inputs)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    legacyInjector = LegacyInjector(model, probability=1.0)
    injector = Injector(model, probability=1.0)

    startOld = time.time()
    legacyInjector.injectError()
    endOld = time.time()

    startNew = time.time()
    injector.injectError()
    endNew = time.time()

    durationOld = endOld - startOld
    durationNew = endNew - startNew

    print(f"duration legacy: {durationOld}, duration new: {durationNew}")
