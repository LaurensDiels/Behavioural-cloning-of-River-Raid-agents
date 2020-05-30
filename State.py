import numpy


class State:

    def __init__(self, screen: numpy.ndarray, is_terminal: bool):
        self.screen = screen
        self.is_terminal = is_terminal
