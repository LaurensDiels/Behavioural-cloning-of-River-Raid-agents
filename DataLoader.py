from abc import ABC, abstractmethod
import os
import numpy
import compress_pickle

from FrameStacker import FrameStacker


class DataLoader(ABC):

    @abstractmethod
    def load_x(self) -> numpy.ndarray:
        pass

    def load_stacked_x(self, nb_to_stack: int, stack_to_last_component: bool) -> numpy.ndarray:
        return FrameStacker.stack_all_frames(self.load_x(), nb_to_stack, stack_to_last_component)

    @abstractmethod
    def load_y(self) -> numpy.ndarray:
        pass


class StateActionDataLoader(DataLoader):
    """Use in conjunction with StateActionSaver."""

    def __init__(self, states_folder: str, actions_folder: str, prefix: str, number: int):
        self.states_file = os.path.join(states_folder, prefix + str(number) + ".npz")
        self.actions_file = os.path.join(actions_folder, prefix + str(number) + ".npz")

    def load_states(self) -> numpy.ndarray:
        loaded = numpy.load(self.states_file)
        return loaded["arr_0"]

    def load_x(self) -> numpy.ndarray:
        return self.load_states()

    def load_actions(self) -> numpy.ndarray:
        loaded = numpy.load(self.actions_file)
        return loaded["arr_0"]

    def load_y(self) -> numpy.ndarray:
        return self.load_actions()


class EpisodeDataLoader(DataLoader):
    """Use in conjunction with EpisodeSaver."""

    def __init__(self, folder: str, prefix: str, number: int):
        self.episode_file = os.path.join(folder, prefix + str(number) + ".xz")
        with open(os.path.join(self.episode_file), "rb") as file:
            self.episode = compress_pickle.load(file, compression="lzma")

    def load_states(self) -> numpy.ndarray:
        return numpy.asarray(list(map(lambda step: step.state.screen, self.episode.steps[:-1])))
        # -1: There is no action based on the last state.

    def load_x(self) -> numpy.ndarray:
        return self.load_states()

    def load_actions(self) -> numpy.ndarray:
        return numpy.asarray(list(map(lambda step: step.action, self.episode.steps[1:])))
        # 1: The zeroth action is None. The first action is based on the initial (zeroth) state.

    def load_y(self) -> numpy.ndarray:
        return self.load_actions()
