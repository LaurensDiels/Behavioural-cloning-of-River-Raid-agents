from abc import ABC, abstractmethod
from random import randint
from typing import Callable

import numpy
import keras
import gym
import keras_gym

from AtariActions import A_FIRST, A_LAST
import KeyboardInput
from Episode import Episode
from KeyboardInput import InputTypes

from FrameStacker import FrameStacker
import Tools
from GlobalSettings import *


def input_quit():
    return KeyboardInput.input_quit()


class AInputHandler(ABC):  # abstract

    @abstractmethod
    def get_action(self, episode: Episode) -> int:
        pass


class HumanInputHandler(AInputHandler):

    def __init__(self, input_type: InputTypes):
        self.input_type = input_type

    def get_action(self, episode: Episode = None) -> int:
        return KeyboardInput.get_input_action(self.input_type)


class RandomInputHandler(AInputHandler):

    def get_action(self, episode: Episode = None, environment=None) -> int:
        if environment:
            return environment.action_space.sample()
        else:
            return randint(A_FIRST, A_LAST)
        # These should be equivalent.


class ModelInputHandler(AInputHandler):

    def __init__(self, screen_to_actions_function: Callable,
                 normalize_to_unit_interval: bool, preprocess: bool,
                 nb_frames: int = 1, stack_in_colors: bool = None):
        """states_to_actions_function (e.g. model.predict (which expects a batch) should map a (possibly preprocessed)
        screen to the relevant action.
        normalize_to_unit_interval converts the pixel values to the interval [0, 1].
        Using preprocess the screen is first preprocessed using the settings in GlobalSettings (which are used
        in the keras_gym reinforcement learner.
        nbFrames allows to use also previous frames."""
        self.screen_to_actions_function = screen_to_actions_function
        self.normalize_to_unit_interval = normalize_to_unit_interval
        self.preprocess = preprocess
        self.nb_frames = nb_frames
        self.stack_in_colors = stack_in_colors

    def get_action(self, episode: Episode) -> int:
        model_input = None
        if self.nb_frames > 1:
            frames = []
            for i in range(max(0, episode.current_length() - self.nb_frames), episode.current_length()):
                frame = episode.steps[i].state.screen
                if self.preprocess:
                    frame = Tools.preprocess_frame(frame)
                if self.normalize_to_unit_interval:
                    frame = frame.astype(NORMALISED_SCREEN_DATA_TYPE) / MAX_PIXEL_VALUE
                frames.append(frame)

            model_input = FrameStacker.stack_last_frame(numpy.array(frames), self.nb_frames, self.stack_in_colors)
        else:
            model_input = episode.current_state().screen
        return self.screen_to_actions_function(model_input)


class KerasModelInputHandler(ModelInputHandler):
    """For models obtained using BatchLearner."""

    def __init__(self, model_path: str, nb_stacked_frames: int = 1, stack_in_colors: bool = False):
        self.model = keras.models.load_model(model_path)
        super().__init__(lambda screen: self.model.predict(numpy.asarray([screen])).argmax(),
                         True, False, nb_stacked_frames, stack_in_colors)


class KerasGymModelInputHandler(ModelInputHandler):
    """To be used in conjunction with KerasGymRL.py."""

    def __init__(self, predict_model_weights_path: str):
        env = gym.make("Riverraid-v0")  # Dummy so that we can make pi below.
        env = keras_gym.wrappers.ImagePreprocessor(env, height=RL_PREPROCESS_HEIGHT, width=RL_PREPROCESS_WIDTH,
                                                   grayscale=RL_PREPROCESS_GRAYSCALE)
        # The actual preprocessing will be done using the preprocess parameter for super().__init__.
        # This way we can take 'normal' screens as input.
        env = keras_gym.wrappers.FrameStacker(env, num_frames=RL_PREPROCESS_NUM_FRAMES)
        env = keras_gym.wrappers.TrainMonitor(env)

        func = keras_gym.predefined.AtariFunctionApproximator(env)
        self.pi = keras_gym.SoftmaxPolicy(func, update_strategy=RL_PI_UPDATE_STRATEGY)
        self.pi.predict_model.load_weights(predict_model_weights_path)

        super().__init__(lambda screen: self.pi(screen), False, True, RL_PREPROCESS_NUM_FRAMES)


class ReplayInputHandler(AInputHandler):
    """Replay episodes saved using StateActionSaver.
    !!! Does not work properly due to the variability of the execution of the environment.
    !!! To watch replays, use WatchReplay.py instead."""

    def __init__(self, action_episode_file_path: str):
        """The file should be created using StateActionSaver and therefore should have the extension ".npz"."""
        loaded = numpy.load(action_episode_file_path)
        self.actions = loaded['arr_0']
        self.step_nb = 0

    def get_action(self) -> int:
        action = self.actions[self.step_nb]
        self.step_nb += 1
        return action
