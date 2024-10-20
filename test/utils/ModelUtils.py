import keras
from keras import layers

class ModelUtils:

    @staticmethod
    def getBasicModel():
        inputs = keras.Input(shape=(37,))
        x = keras.layers.Dense(32, activation="relu")(inputs)
        outputs = keras.layers.Dense(5, activation="softmax")(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    @staticmethod
    def getSequentialModel():
        return keras.Sequential(
            [
                layers.Dense(2, activation="relu", name="layer1"),
                layers.Dense(3, activation="relu", name="layer2"),
                layers.Dense(4, name="layer3"),
            ]
        )
