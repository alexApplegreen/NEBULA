import struct

from NEBULA.core.noiseLayer import NoiseLayer

from tensorflow.keras.models import load_model


def binary(num):
    """
    Helper Function to display the binary representation of floating point numbers
    Source: https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
    Validated using https://www.h-schmidt.net/FloatConverter/IEEE754.html
    """
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))


def loadFatModel(path: str, *args, **kwargs):
    """Load Fault-Aware-Trained model using the NEBULA.traininginjector from .h5 file
    This passes all arguments through to the load_model function from keras,
    while supplying the underlying NoiseLayer NEBULA uses for fault aware training.
    This layer must be known for Keras in order to load the model correctly.
    """
    if 'custom_objects' not in kwargs:
        kwargs['custom_objects'] = {'NoiseLayer': NoiseLayer}
    return load_model(path, *args, **kwargs)