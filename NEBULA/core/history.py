#!/usr/bin/env python3

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

import copy
from collections import deque

from keras import Layer


class History(deque):
    """History function for injector classes
    Subclasses the native deque class, acts like a stack for Model instances.
    Adheres to standard FIFO stack implementations
    """

    def __init__(self, layers: list[Layer] = []) -> None:
        """Initializes an empty history

        Parameters:
            layers (list[Layer]): Layers of the model under test
        """
        super().__init__()
        if layers is not []:
            layerCopy = copy.deepcopy(layers)
            self.push(layerCopy)

    def push(self, entry: list[Layer]) -> None:
        """Add an element to the history

        Parameters:
            entry (list[Layer]): A full set of layers of the model under test
        """
        layerCopy = copy.deepcopy(entry)
        self.append(layerCopy)

    def revert(self) -> None:
        """ Revert last change made to history
        Removes the element from top of the stack
        """
        try:
            super().pop()
        except IndexError:
            raise IndexError("pop from an empty history")

    def pop(self) -> list[Layer]:
        """Retrieves a set of Layers from the top of the history
        This method removes the first element from the history

        Returns:
            list[Layer]: The full set of layers that was saved last
        """
        try:
            return super().pop()
        except IndexError:
            raise IndexError("pop from an empty history")

    def peek(self) -> list[Layer]:
        """Retrieves a set of layers from the top of the history
        This method does not remove the set of layers from the top of the history

        Returns:
            list[Layer]: The full set of layers that was saved last
        """
        try:
            elem = super().pop()
            super().append(elem)
            return elem
        except IndexError:
            raise IndexError("peek from an empty history")

    def size(self) -> int:
        """Gives the size of the history
        Returns:
            int: The size (length) of the history
        """
        return len(self)
