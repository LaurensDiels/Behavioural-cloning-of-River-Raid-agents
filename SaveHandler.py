import threading
import time

import numpy
import os
import compress_pickle
from abc import ABC, abstractmethod

from Episode import Episode


class SaveHandler(ABC):

    def __init__(self, save_interrupted_episodes: bool, nb_save_threads: int = None):
        """When nb_save_threads is None it will be given the default value of 2."""
        if nb_save_threads is None:
            nb_save_threads = 2
        self.save_interrupted_episodes = save_interrupted_episodes
        self.nb_save_threads = nb_save_threads
        self.save_threads = [None] * nb_save_threads
        self.current_thread_nb = -1

    def handle_episode(self, episode: Episode, verbose: bool = False):
        """If verbose is true we will print a statement that we need to wait for a thread to finish saving a previous
        episode (but note that this wait might be over immediately), as well as a statement when we start saving the
        current episode."""
        if not self.will_save_episode(episode):
            return
        else:
            self.current_thread_nb += 1
            if self.current_thread_nb >= self.nb_save_threads:
                self.current_thread_nb = 0

            if self.save_threads[self.current_thread_nb]:
                if verbose:
                    print("Waiting for a previous episode to save...")
                self.save_threads[self.current_thread_nb].join()

            self.save_threads[self.current_thread_nb] = threading.Thread(target=self._save_episode, args=(episode,))
            self.save_threads[self.current_thread_nb].start()

            if verbose:
                print("Saving current episode in the background.")
        pass

    @abstractmethod
    def will_save_episode(self, episode: Episode):
        pass

    @abstractmethod
    def _save_episode(self, episode: Episode):
        pass

    def finish_all_save_threads(self):
        for t in self.save_threads:
            if t is not None:
                t.join()


class NoSaver(SaveHandler):

    def __init__(self):
        super().__init__(False)

    def will_save_episode(self, episode: Episode):
        return False

    def _save_episode(self, episode: Episode):
        return


class StateActionSaver(SaveHandler):
    """Used to save episodes. Saves the state (pixels) - action pairs, where the chosen action is based on the state
        (and thus state_i is NOT the result of the action_i, as in Step).
        This configuration makes sense for inverse reinforcement learning.

        One file will contain the states, another file with the same number the actions."""

    def __init__(self, save_interrupted_episodes: bool, states_folder: str, actions_folder: str, prefix: str,
                 nb_save_threads: int = None):
        """Expects the supplied folders to either not exist, or be created using StateActionSaver."""
        super().__init__(save_interrupted_episodes, nb_save_threads)
        self.states_folder = states_folder
        self.actions_folder = actions_folder
        self.prefix = prefix
        self.save_interrupted_episodes = save_interrupted_episodes

        # Make dirs, if necessary. Note that if states_folder = "", it already exists.
        if states_folder and not os.path.exists(states_folder):
            os.makedirs(states_folder)
            # In contrast to mkdir, mkdirs will also create parent directories, if those do not exist.
        if actions_folder and not os.path.exists(actions_folder):
            os.makedirs(actions_folder)

        # Count the number of files in states_folder.
        # If these folders were created by StateActionSaver, actions_folder contains as many files.
        self.nb_saves = len(os.listdir(states_folder))

    def _save_episode(self, episode: Episode):
        file_nb = self.nb_saves
        self.nb_saves += 1  # Update as soon as possible, since other threads might need it before we are finished.

        # A current action is based on the previous state. We need (Step) pairs (state_i, action_{i + 1}.
        states = numpy.asarray(list(map(lambda step: step.state.screen, episode.steps[:-1])))
        actions = numpy.asarray(list(map(lambda step: step.action, episode.steps[1:])))

        # Save compressed to save on disk space. (Saving per episode instead of per pair also helps in this regard.)
        # os.path.join: to get the proper slash ('\' or '/')
        # numpy.savez_compressed automatically adds the extension .npz
        numpy.savez_compressed(os.path.join(self.states_folder, self.prefix + str(file_nb)), states)
        numpy.savez_compressed(os.path.join(self.actions_folder, self.prefix + str(file_nb)), actions)

    def will_save_episode(self, episode: Episode):
        return episode.finished() or self.save_interrupted_episodes


# Choice for LZMA in compress_pickle:
#
# Average timings over five runs (sizes always remains constant) on an i7 7700k (and ssd):
# (always same episode)
#   Compression:
#       No compression:               182 915 kB;    189 ms
#       gzip:                           1 787 kB;  6 106 ms -- compress_pickle gzip:          1 726 kB;    5 955 ms
#       mgzip:                          1 785 kB;  3 838 ms  [mgzip = multithreaded gzip]
#       bzip2:                            317 kB; 26 008 ms -- compress_pickle bzip2:           265 kB;   27 131 ms
#       LZMA:                             231 kB; 23 751 ms -- compress_pickle LZMA:            197 kB;   23 480 ms
#       LZ4:                            8 149 kB;    189 ms
#                                                           -- compress_pickle zipfile:     182 860 kB;      309 ms
#       numpy.save:                   182 602 kB;    501 ms
#       numpy.savez_compressed:         1 938 kB;  1 397 ms
#   Decompression:
#       No compression:                              141 ms
#       gzip:                                        370 ms -- compress_pickle gzip:                         369 ms
#       mgzip:                                       646 ms
#       bzip2:                                     2 886 ms -- compress_pickle bzip2:                      3 068 ms
#       LZMA:                                        640 ms -- compress_pickle LZMA:                         623 ms
#       LZ4:                                         187 ms
#                                                           -- compress_pickle zipfile:                      423 ms
#       numpy.save:                                   83 ms
#       numpy.savez_compressed:                      393 ms
#
# using code of the form
#     with gzip.GzipFile('gzip.gz', 'wb') as f:
#         pickle.dump(episode, f)
# and
#     with open('CPgzip.gz', 'wb') as f:
#         compress_pickle.dump(episode, f, compression="gzip")
#
# For the numpy methods we only saved the states, not the entire Episode.
# (Here 1 kB = 1024 bytes.)
# (The baseline does vary a lot from episode to episode, but the relative figures remained relatively the same.)

class EpisodeSaver(SaveHandler):
    """Used to save full episodes compressed using LZMA (internally uses compress_pickle.dump)."""

    def __init__(self, save_interrupted_episodes: bool, folder: str, prefix: str, start_file_nb: int = None,
                 nb_save_threads: int = None):
        """Expects the supplied folder to either not exist, or be created using EpisodeDumpSaver."""
        super().__init__(save_interrupted_episodes, nb_save_threads)
        self.folder = folder
        self.prefix = prefix

        # Make directory, if necessary.
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            # In contrast to mkdir, mkdirs will also create parent directories, if those do not exist.

        if start_file_nb:
            self.file_nb = start_file_nb
        else:
            # Count the number of files in folder.
            self.file_nb = len(os.listdir(folder))

    def _save_episode(self, episode: Episode):
        current_file_nb = self.file_nb
        self.file_nb += 1

        with open(os.path.join(self.folder, self.prefix + str(current_file_nb) + ".xz"), "wb") as file:
            compress_pickle.dump(episode, file, compression="lzma")

    def will_save_episode(self, episode: Episode):
        return episode.finished() or self.save_interrupted_episodes
