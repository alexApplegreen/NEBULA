#!/usr/bin/env python3

"""
history.py
    provides utility to undo changes made by the errorinjector
"""

__author__      = "Alexander Tepe"
__email__       = "alexander.tepe@hotmail.de"
__copyright__   = "Copyright 2024, Planet Earth"

import copy
from collections import deque

from keras import Layer, Model


class History(deque):
    """Subclass of the deque class
    acts like a stack for Model instances.
    Interface is as you would expect:
        push - add element to top of stack
        pop - get and remove from top of stack
        peek - get element from top without removing it
    """

    def __init__(self, layers: list[Layer]=[]) -> None:
        super().__init__()
        if layers is not []:
            layerCopy = copy.deepcopy(layers)
            self.push(layerCopy)

    def push(self, entry: list[Layer]) -> None:
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
        try:
            return super().pop()
        except IndexError:
            raise IndexError("pop from an empty history")

    def peek(self) -> list[Layer]:
        try:
            elem = super().pop()
            super().append(elem)
            return elem
        except IndexError:
            raise IndexError("peek from an empty history")

    def size(self):
        return len(self)
