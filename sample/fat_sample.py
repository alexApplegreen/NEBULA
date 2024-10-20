from NEBULA import TrainingInjector

import keras


if __name__ == '__main__':
    inputs = keras.Input(shape=(37,))
    x = keras.layers.Dense(32, activation="relu")(inputs)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    ti = TrainingInjector()
    newModel = ti.attach(model, index=2)

    newModel.summary()
