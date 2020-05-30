import gym
import time
import keyboard
import numpy
import sys

from GlobalSettings import *

from InputHandler import HumanInputHandler, RandomInputHandler, KerasModelInputHandler, KerasGymModelInputHandler
from VisualizationHandler import NoVisualizer, NoSleepVisualizer, MaxFPSVisualizer
from Episode import Episode
from State import State
from KeyboardInput import input_quit, InputTypes
from KeyboardScanCodes import KEY_ENTER
from SaveHandler import NoSaver, StateActionSaver, EpisodeSaver
from AtariActions import atari_action_to_name, A_NOOP, A_DOWN


def main_RR():

    env = gym.make("Riverraid-v0")

    ##############
    # Settings:
    SAVE_INTERRUPTED_EPISODES = False

    # SAVE_EPISODES_FOLDER = HUMAN_EPISODES_FOLDER
    # SAVE_EPISODES_FOLDER = FIXED_RL_EPISODES_FOLDER
    SAVE_EPISODES_FOLDER = RANDOM_EPISODES_FOLDER

    saver = NoSaver()
    # saver = StateActionSaver(SAVE_INTERRUPTED_EPISODES, STATES_FOLDER, ACTION_FOLDER, EPISODE_PREFIX)
    # saver = EpisodeSaver(SAVE_INTERRUPTED_EPISODES, SAVE_EPISODES_FOLDER, EPISODE_PREFIX, nb_save_threads=4)

    input_handler = HumanInputHandler(InputTypes.COMBINED)
    # input_handler = RandomInputHandler()
    # input_handler = KerasModelInputHandler(HUMAN_BC_CUSTOM_A_MODEL_TL_FINETUNE_MODEL_PATH,
    #                                       CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_NB_FRAMES_TO_STACK, True)
    # input_handler = KerasGymModelInputHandler(FIXED_RL_PI_PREDICT_MODEL_WEIGHTS_PATH)

    # visualization_handler = MaxFPSVisualizer(env, ATARI_FPS)
    visualization_handler = NoSleepVisualizer(env)
    # visualization_handler = NoVisualizer()

    # WAIT_TO_START = True
    WAIT_TO_START = False
    START_KEY = KEY_ENTER
    WAIT_TO_START_SLEEP_TIME = 0.1  # i.e. 100ms

    FORCE_START = False  # Useful for when models decide to continually spam noops at the start.
    ##############

    stop = False
    while not stop:
        initial_observation = env.reset()
        episode = Episode(State(initial_observation, False))

        if WAIT_TO_START:
            while not keyboard.is_pressed(START_KEY):
                time.sleep(WAIT_TO_START_SLEEP_TIME)
                visualization_handler.handle_visualization()
                if input_quit():
                    stop = True
                    break
        if stop:
            break

        # Complete one episode
        started = False
        while not (episode.finished() or stop):
            visualization_handler.set_start()

            action = input_handler.get_action(episode)
            print("Action: {}".format(atari_action_to_name(action)))
            if not started and action == A_NOOP and FORCE_START:
                action = A_DOWN  # The non-noop action with the least impact

            observation, reward, done, _ = env.step(action)
            # Note: if we replace _ with info, we can also get the remaining number of lives via info["ale.lives"].
            # But, as per OpenAI's rules, agents are not allowed to use this information (or anything from info).
            started = not numpy.array_equal(observation, episode.current_state().screen)
            # Note that every time you die, the game 'pauses'.

            episode.add_step(action, reward, State(observation, done))
            visualization_handler.handle_visualization()
            stop = input_quit()
            # Note: all keyboard.is_pressed calls will cause the program to crash on Linux when not root.

        print("Episode score: {}".format(episode.get_score()))
        print("")

        # Save
        saver.handle_episode(episode, verbose=True)

    env.close()  # Note: sometimes doesn't close but freezes instead.
    saver.finish_all_save_threads()


if __name__ == "__main__":
    main_RR()
    sys.exit()

