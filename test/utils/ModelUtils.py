import keras

class ModelUtils:

    @staticmethod
    def getBasicModel():
        inputs = keras.Input(shape=(37,))
        x = keras.layers.Dense(32, activation="relu")(inputs)
        outputs = keras.layers.Dense(5, activation="softmax")(x)
        return keras.Model(inputs=inputs, outputs=outputs)