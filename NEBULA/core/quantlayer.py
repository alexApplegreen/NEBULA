from keras import Layer

class QuantLayer(Layer):
    """Layer subclass which supports arbitrary datatypes
    """
    # TODO build this

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.autocast = False

    def build(self, input_shape):
        # TODO
        pass

    def call(self):
        # TODO
        pass
