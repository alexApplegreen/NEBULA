from threading import Thread
from keras.src.models import Model


def injectToWeights(model: Model, probability: float) -> Model:
    """modify weights of model in place using concurrency
    """
    for layer in model.layers:
        thread = Thread(
            target=_concurrentErrorInjection,
            args=(layer, probability)
        )
        thread.start()
        thread.join()

    return model

def _concurrentErrorInjection(layer, probability):
    weights = layer.get_weights()
    for weight in weights:
        shape = weight.shape
        flattened = weight.flatten()
        for i in range(len(flattened)):
            # TODO actually flip bits
            newWeight = flattened.reshape(shape)
            layer.set_weights(newWeight)
