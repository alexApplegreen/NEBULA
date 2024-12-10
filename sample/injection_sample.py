import keras

from NEBULA import Injector, ErrorTypes

if __name__ == '__main__':

    # Simple, generic model
    inputs = keras.Input(shape=(37,))
    x = keras.layers.Dense(32, activation="relu")(inputs)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # create injector
    injector = Injector(model.layers, probability=1.0)

    # inject binomial distributed error
    injector.injectError(model, ErrorTypes.NORMAL)

    # reset error
    injector.undo(model)
