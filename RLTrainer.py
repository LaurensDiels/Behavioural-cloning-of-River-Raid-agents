import threading
from abc import ABC, abstractmethod

import glob
import gym
import keras_gym as km
import numpy

from Episode import Episode
from SaveHandler import EpisodeSaver
from State import State

from GlobalSettings import *


class RLTrainer(ABC):
    """Abstract base class for training on Riverraid using keras_gym's PPO
    (https://keras-gym.readthedocs.io/en/stable/notebooks/atari/ppo.html).
    Although in its initialization we take in and save variables concerning saving, and provide saving methods,
    the actual saving is left to subclasses via handle_end_of_episode.
    """

    GAMMA = 0.99
    BOOTSTRAP_WITH_TARGET_MODEL = True
    BOOTSTRAP_N = 10
    BUFFER_CAPACITY = 256
    BUFFER_BATCH_SIZE = 64
    ACTOR_CRITIC_UPDATE_TAU = 0.1

    def __init__(self, base_folder: str, models_folder_name: str,
                 load_pi_predict_model: bool, load_v_predict_model: bool,
                 save_episodes: bool, save_episodes_folder: str,
                 save_gifs: bool, gifs_folder_name: str,
                 save_pi_predict_models: bool, save_v_predict_models: bool,
                 run_indefinitely: bool, max_nb_episodes: int,
                 use_keras_gym_train_monitor: bool):
        """It makes little sense to have both save_episodes and save_gifs set to True, since episodes can be watched
        (in better quality, even though the files are smaller) using WatchReplay.py."""

        self.base_folder = base_folder
        self.models_folder_name = models_folder_name
        self.save_episodes = save_episodes
        self.save_episodes_folder = save_episodes_folder
        self.should_save_gifs = save_gifs
        self.gifs_folder_name = gifs_folder_name
        self.should_save_pi_predict_models = save_pi_predict_models
        self.should_save_v_predict_models = save_v_predict_models
        self.run_indefinitely = run_indefinitely
        self.max_nb_episodes = max_nb_episodes
        self.use_keras_gym_train_monitor = use_keras_gym_train_monitor

        self.models_folder = os.path.join(self.base_folder, self.models_folder_name)
        self.gifs_folder = os.path.join(self.base_folder, self.gifs_folder_name)

        if save_episodes_folder and not os.path.exists(save_episodes_folder):
            os.makedirs(save_episodes_folder)
        if models_folder_name and not os.path.exists(self.models_folder):
            os.makedirs(self.models_folder)
        if save_gifs and self.gifs_folder and not os.path.exists(self.gifs_folder):
            os.makedirs(self.gifs_folder)

        self.env = gym.make('Riverraid-v0')
        self.env = km.wrappers.ImagePreprocessor(self.env, height=RL_PREPROCESS_HEIGHT, width=RL_PREPROCESS_WIDTH,
                                                 grayscale=RL_PREPROCESS_GRAYSCALE)
        self.env = km.wrappers.FrameStacker(self.env, num_frames=RL_PREPROCESS_NUM_FRAMES)
        if use_keras_gym_train_monitor:
            self.env = km.wrappers.TrainMonitor(self.env)

        # show logs from TrainMonitor
        km.enable_logging()

        # function approximators
        self.func = km.predefined.AtariFunctionApproximator(self.env)
        self.pi = km.SoftmaxPolicy(self.func, update_strategy=RL_PI_UPDATE_STRATEGY)  # PPO

        self.v = km.V(self.func, gamma=RLTrainer.GAMMA,
                      bootstrap_with_target_model=RLTrainer.BOOTSTRAP_WITH_TARGET_MODEL,
                      bootstrap_n=RLTrainer.BOOTSTRAP_N)

        self.actor_critic = km.ActorCritic(self.pi, self.v)

        # we'll use this to temporarily store our experience
        self.buffer = km.caching.ExperienceReplayBuffer.from_value_function(
            value_function=self.v, capacity=RLTrainer.BUFFER_CAPACITY, batch_size=RLTrainer.BUFFER_BATCH_SIZE)

        if load_pi_predict_model:
            self.load_pi_predict_model_weights()
        if load_v_predict_model:
            self.load_v_predict_model_weights()

    def load_pi_predict_model_weights(self):
        """Loads the last saved pi.predict_model weights (with the highest episode number suffix).
        (Also loading pi.target_model actually gives worse results (see documentation load_v_model).)"""
        prefix = os.path.join(self.models_folder, RL_PI_PREDICT_MODEL_WEIGHTS_BASE_FILE_NAME)
        saved_pi_weights_files = glob.glob(prefix + "*.h5")
        if saved_pi_weights_files:  # (there might be no saved weights at all)
            # The saved model weights files are of the form prefix + str(nb_episodes) + ".h5".
            self.env.ep = max([int(x[len(prefix):-3]) for x in saved_pi_weights_files])  # len(".h5") == 3
            pi_predict_model_weights_path = "{}{}.h5".format(prefix, self.env.ep)
            self.pi.predict_model.load_weights(pi_predict_model_weights_path)

    def load_v_predict_model_weights(self):
        """Loads the last saved v.predict_model weights (with the highest episode number suffix).
        (Also loading v.target_model actually gives worse results.)

        The best results are obtained when loading both pi and v. If only loading one of them, pi is the most important,
        v does not seem to do much on its own.
        Some test results: (loading an RL model with scores around 1800 - 2000)
        Training 50 episodes from scratch: average score 1570
        pi and v: 1820, 2090 (2 tests)
        just pi:  1750, 1800
        just v:   1560
        Note: this is just loading the predict_models. When also loading target_model (using the predict weights)
        for both we got 1490. (Loading train_model leads to crashes.)
        """
        prefix = os.path.join(self.models_folder, RL_V_PREDICT_MODEL_WEIGHTS_BASE_FILE_NAME)
        saved_v_weights_files = glob.glob(prefix + "*.h5")
        if saved_v_weights_files:
            self.env.ep = max([int(x[len(prefix):-3]) for x in saved_v_weights_files])
            v_predict_model_weights_path = "{}{}.h5".format(prefix, self.env.ep)
            self.v.predict_model.load_weights(v_predict_model_weights_path)

    def train(self):
        while self.run_indefinitely or self.env.ep < self.max_nb_episodes:
            if not self.use_keras_gym_train_monitor:
                self.env.ep += 1  # Apparently this update is done by the train monitor
            episode = self.run_for_episode()
            self.handle_end_of_episode(episode)
        self.finish()

    def run_for_episode(self) -> Episode:
        s = self.env.reset()
        s_orig = self.env.unwrapped.reset()
        episode = Episode(State(s_orig, False))
        for t in range(self.env.spec.max_episode_steps):
            a = self.pi(s, use_target_model=True)
            s_next, r, done, info = self.env.step(a)

            self.buffer.add(s, a, r, done, self.env.ep)

            if len(self.buffer) >= self.buffer.capacity:
                # use 4 epochs per round
                num_batches = int(4 * self.buffer.capacity / self.buffer.batch_size)
                for _ in range(num_batches):
                    self.actor_critic.batch_update(*self.buffer.sample())
                self.buffer.clear()

                # soft update (tau=1 would be a hard update)
                self.actor_critic.sync_target_model(tau=RLTrainer.ACTOR_CRITIC_UPDATE_TAU)

            s = s_next
            s_orig = info["s_next_orig"][0]

            episode.add_step(a, r, State(s_orig, done))

            if done:
                break

        return episode

    @abstractmethod
    def handle_end_of_episode(self, episode: Episode):
        pass

    @abstractmethod
    def finish(self):
        pass

    def save_pi_predict_model_weights(self):
        self.pi.predict_model.save_weights(os.path.join(
            self.models_folder, RL_PI_PREDICT_MODEL_WEIGHTS_BASE_FILE_NAME + str(self.env.ep) + ".h5"))

    def save_v_predict_model_weights(self):
        self.v.predict_model.save_weights(os.path.join(
            self.models_folder, RL_V_PREDICT_MODEL_WEIGHTS_BASE_FILE_NAME + str(self.env.ep) + ".h5"))

    def save_gif(self):
        km.utils.generate_gif(
            env=self.env,
            policy=self.pi,
            filepath=os.path.join(self.gifs_folder, "ep{:06d}.gif".format(self.env.ep)),
            resize_to=(320, 420))


class SimpleRLTrainer(RLTrainer):

    def __init__(self,
                 load_pi_predict_model: bool, load_v_predict_model: bool,
                 save_episodes: bool,
                 save_gifs: bool,
                 save_pi_predict_models: bool, save_v_predict_models: bool,
                 max_nb_episodes: int,
                 base_folder: str = IMPROVING_SIMPLE_RL_BASE_FOLDER,
                 models_folder_name: str = RL_MODEL_FOLDER_NAME,
                 save_episodes_folder: str = IMPROVING_SIMPLE_RL_EPISODES_FOLDER,
                 gifs_folder: str = RL_GIFS_FOLDER_NAME,
                 save_gifs_and_models_starting_episode_nb: int = 0,
                 save_gifs_and_models_frequency: int = 100,
                 run_indefinitely: bool = False,
                 use_keras_gym_train_monitor: bool = True):

        super().__init__(base_folder, models_folder_name,
                         load_pi_predict_model, load_v_predict_model,
                         save_episodes, save_episodes_folder,
                         save_gifs, gifs_folder,
                         save_pi_predict_models, save_v_predict_models,
                         run_indefinitely, max_nb_episodes,
                         use_keras_gym_train_monitor)

        self.save_gifs_and_models_starting_episode_nb = save_gifs_and_models_starting_episode_nb
        self.save_gifs_and_models_frequency = save_gifs_and_models_frequency

        self.episode_scores = []
        self.episode_saver = EpisodeSaver(save_interrupted_episodes=False,
                                          folder=self.save_episodes_folder,
                                          prefix=EPISODE_PREFIX,
                                          start_file_nb=self.env.ep+1,  # (starting from 1)
                                          nb_save_threads=4)

    def handle_end_of_episode(self, episode: Episode):
        self.episode_scores.append(episode.get_score())

        if self.save_episodes:
            self.episode_saver.handle_episode(episode)

        if self.env.ep % self.save_gifs_and_models_frequency == 0 \
                and self.env.ep >= self.save_gifs_and_models_starting_episode_nb:
            if self.should_save_pi_predict_models:
                self.save_pi_predict_model_weights()
            if self.should_save_v_predict_models:
                self.save_v_predict_model_weights()
            if self.should_save_gifs:
                self.save_gif()

    def finish(self):
        self.episode_saver.finish_all_save_threads()


class GreedyRLTrainer(RLTrainer):
    """Greedy trains using PPO as follows. Starting from initial pi and v predict weights, we run PPO for a number of
    episodes. If the average score has improved, we continue. If not we reload the previous best pi and v predict
    weights.

    This will only work if we start from decent initial pi and v weights, obtained using e.g. SimpleRLTrainer.

    (In the automatic output the avg_G s will be incorrect.)
    """

    def __init__(self,
                 initial_agent_score_estimate: float,
                 save_episodes: bool,
                 save_gifs: bool,
                 max_nb_episodes: int,
                 base_folder: str = IMPROVING_GREEDY_RL_BASE_FOLDER,
                 models_folder_name: str = RL_MODEL_FOLDER_NAME,
                 save_episodes_folder: str = IMPROVING_GREEDY_RL_EPISODES_FOLDER,
                 save_temp_episodes_folder_name: str = "Temp",
                 gifs_folder: str = RL_GIFS_FOLDER_NAME,
                 training_block_nb_episodes: int = 100,
                 run_indefinitely: bool = False,
                 verbose: int = 1):
        """With verbose=0 there will be no prints (not recommended), verbose=1 will show custom output: episode numbers,
        episode scores, score thresholds, model saves, etc., and verbose=2 will do the same, but additionally use
        keras_gym's train monitor. (A warning: here the avg_G s will be incorrect as they will keep averaging over all
        episodes, also the discarded ones."""

        super().__init__(base_folder, models_folder_name,
                         True, True,
                         save_episodes, save_episodes_folder,
                         save_gifs, gifs_folder,
                         True, True,
                         run_indefinitely, max_nb_episodes,
                         verbose >= 2)

        self.print_outputs = verbose >= 1
        self.save_temp_episodes_folder_name = save_temp_episodes_folder_name
        self.training_block_nb_episodes = training_block_nb_episodes

        self.save_temp_episodes_folder = os.path.join(self.save_episodes_folder, self.save_temp_episodes_folder_name)
        if save_episodes and not os.path.exists(self.save_temp_episodes_folder):
            os.makedirs(self.save_temp_episodes_folder_name)

        if initial_agent_score_estimate:
            self.last_training_block_average_score = initial_agent_score_estimate
        else:
            self.last_training_block_average_score = self.get_current_agent_average_score()

        self.past_training_blocks_episode_scores = []
        self.current_training_block_episode_scores = []

        self.episode_saver = EpisodeSaver(save_interrupted_episodes=False,
                                          folder=self.save_temp_episodes_folder,
                                          prefix=EPISODE_PREFIX,
                                          start_file_nb=self.env.ep+1,  # (starting from 1)
                                          nb_save_threads=4)
        self.episode_mover_thread = None

    def get_current_agent_average_score(self, nb_episodes_to_test: int = 10):
        if self.print_outputs:
            print("Estimating the current agent's average score...")

        temp_env = gym.make('Riverraid-v0')
        temp_env = km.wrappers.ImagePreprocessor(temp_env, height=RL_PREPROCESS_HEIGHT, width=RL_PREPROCESS_WIDTH,
                                                 grayscale=RL_PREPROCESS_GRAYSCALE)
        temp_env = km.wrappers.FrameStacker(temp_env, num_frames=RL_PREPROCESS_NUM_FRAMES)

        scores = []
        for i in range(nb_episodes_to_test):
            s = temp_env.reset()
            score = 0.0
            for _ in range(self.env.spec.max_episode_steps):
                a = self.pi(s, use_target_model=False)
                s, r, done, info = temp_env.step(a)
                score += r
                if done:
                    break
            scores.append(score)
            if self.print_outputs:
                print("    Episode {}/{}: {}".format(i + 1, nb_episodes_to_test, score))

        temp_env.close()
        avg_score = numpy.mean(scores)
        if self.print_outputs:
            print("Found an average score of {:.2f}.".format(avg_score))
            print("")
        return numpy.mean(avg_score)

    def handle_end_of_episode(self, episode: Episode):
        self.current_training_block_episode_scores.append(episode.get_score())
        current_training_block_average_score = numpy.mean(self.current_training_block_episode_scores)
        # (This could be computed slightly more efficiently by saving and reusing the previous average. However,
        # this should hardly make any difference.)
        if self.print_outputs:
            print("Episode {} - score {} - block_avg {:.2f}".format(self.env.ep,
                                                                    episode.get_score(),
                                                                    current_training_block_average_score)
                  )

        if self.save_episodes:
            self.episode_saver.handle_episode(episode)

        if self.env.ep % self.training_block_nb_episodes == 0:
            if current_training_block_average_score > self.last_training_block_average_score:
                if self.print_outputs:
                    print("Average training block score has improved ({:.2f} -> {:.2f}). "
                          "Saving current weights.".format(self.last_training_block_average_score,
                                                           current_training_block_average_score)
                          )

                self.past_training_blocks_episode_scores.extend(self.current_training_block_episode_scores)
                self.last_training_block_average_score = current_training_block_average_score

                self.save_pi_predict_model_weights()
                self.save_v_predict_model_weights()

                if self.save_episodes:
                    if self.print_outputs:
                        print("Moving this training block's episodes from the temp folder to the main one.")
                    self.move_saved_episodes_from_temp(self.episode_saver)
                    # Create a new EpisodeSaver, so that we can wait for the previous one to finish.
                    self.episode_saver = EpisodeSaver(save_interrupted_episodes=False,
                                                      folder=self.save_temp_episodes_folder,
                                                      prefix=EPISODE_PREFIX,
                                                      start_file_nb=self.env.ep+1,
                                                      nb_save_threads=4)
            else:
                if self.print_outputs:
                    print("Average training block score has not improved ({:.2f} -> {:.2f}). "
                          "Reloading last saved weights.".format(self.last_training_block_average_score,
                                                                 current_training_block_average_score)
                          )

                # Load last saved weights (also changes self.env.ep)
                self.load_pi_predict_model_weights()
                self.load_v_predict_model_weights()
                self.episode_saver.file_nb = self.env.ep + 1

            self.current_training_block_episode_scores = []
            if self.should_save_gifs:
                self.save_gif()

    def move_saved_episodes_from_temp(self, old_episode_saver: EpisodeSaver):
        if self.episode_mover_thread:
            self.episode_mover_thread.join()
        self.episode_mover_thread = threading.Thread(target=self.move_saved_episodes_from_temp_target,
                                                     args=(old_episode_saver,))
        self.episode_mover_thread.start()

    def move_saved_episodes_from_temp_target(self, old_episode_saver: EpisodeSaver):
        """Move the last self.training_block_nb_episodes from temp to the main episode save folder."""
        old_episode_saver.finish_all_save_threads()
        for i in range(old_episode_saver.file_nb - self.training_block_nb_episodes, old_episode_saver.file_nb):
            # file_nb is the next file to save (e.g. 1101 instead of 1100 <-- last saved)
            file_name = EPISODE_PREFIX + str(i) + ".xz"
            os.rename(os.path.join(self.save_temp_episodes_folder, file_name),
                      os.path.join(self.save_episodes_folder, file_name))

    def finish(self):
        self.episode_saver.finish_all_save_threads()
        self.episode_mover_thread.join()