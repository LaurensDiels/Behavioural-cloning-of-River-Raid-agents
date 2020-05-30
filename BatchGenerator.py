from typing import List
import numpy
import threading
import time
import compress_pickle
import copy

from AtariActions import NB_ACTIONS

from GlobalSettings import *

PRINT_DEBUG_PREFIX = "|\t"
NB_CLASSES = NB_ACTIONS


class BatchGenerator:
    """Used to sequentially load saved data and return batches."""

    def __init__(self, folder: str, prefix: str, file_nbs: List[int], data_loader_class,
                 batch_size: int, nb_epochs: int = 1,
                 nb_stacked_data: int = None, stack_in_last_component: bool = True,
                 allow_incomplete_last_batch: bool = True, reset_when_done: bool = False,
                 wrap_back_after_epoch: bool = True,
                 nb_load_threads: int = 2,
                 step: int = 1, offset: int = 0,
                 print_debug: bool = False):
        """
        folder and prefix together with the values in file_nbs are the arguments used to load files using a DataLoader.
        data_loader_class has to be a (non-abstract) subclass of DataLoader.

        If nb_stacked_data is set (to something other than None) we will load stacked data, i.e. call
        data_loader_class.load_stacked_x instead of data_loader_class.load_x. Obviously, nb_stacked_data then refers
        to the number of input points to stack. With stack_in_last_component set to False, the stacking will occur
        in a new dimension (the new first), otherwise we stack in the last component. For example, if the data are
        color images of shape (x, y, 3) and we stack them per 4, then the stacked data have shape (x, y, 12). This
        can be more convenient if the batches will be fed to a CNN.

        If wrap_back_after_epoch is True, then the generator will not stop at the last file.
        Instead it will start again from the first file. Therefore, if allow_incomplete_last_batch is false,
        there might be a batch containing data from both the last and the first file.
        With wrap_back_after_epoch set to True, the generator will stop when nb_epochs have expired.

        If reset_when_done, the generator will never stop. Instead, when it has finished (reached the last file,
        or the next_batch_epoch limit), it will reset, and start again from the beginning.

        Via step you can skip, say, half of the data points. Offset determines the first data point to be included
        (where we start counting at the zeroth entry of first_file_nb). When using (very) high values for skip,
        higher values for nb_load_threads could be useful.

        When print_debug is True execution time metrics will be printed. These start with a number of times
        PRINT_DEBUG_PREFIX, where this number depends on the internal debug depth.

        To work in combination with keras.fit_generator(), use allow_incomplete_last_batch = False,
        reset_when_done = True and wrap_back_after_epoch = False.
        """
        self.folder: str = folder
        self.prefix: str = prefix
        self.batch_size: int = batch_size
        self.allow_incomplete_last_batch: bool = allow_incomplete_last_batch
        self.wrap_back_after_epoch: bool = wrap_back_after_epoch
        self.next_batch_epoch: int = 1  # Starts at 1. The next batch will be in this epoch.
        self.last_batch_epoch: int = 0  # The lastly generated batch was in this epoch.
        self.nb_epochs: int = nb_epochs
        self.reset_when_done: bool = reset_when_done

        self.print_debug: bool = print_debug
        self.batch_generation_start_time: float = 0.0
        self.batch_generation_time: float = 0.0
        self.went_to_next_file: bool = False
        self.wait_for_file_to_load_time: float = 0.0

        self.data_loader_class = data_loader_class
        self.nb_stacked_data: int = nb_stacked_data
        self.stack_in_last_component: bool = stack_in_last_component
        self.file_nbs: List[int] = file_nbs

        self.step: int = step
        self.offset: int = offset

        self.current_file_index: int = 0
        self.current_in_file_entry: int = self.offset
        self.current_load_file_index: int = 0
        self.current_file_thread_nb: int = 0
        self.last_batch_file_index: int = -1

        self.nb_load_threads: int = min(nb_load_threads, len(self.file_nbs))
        # No need for more loading threads than files to load.
        self.load_threads: List[threading.Thread] = [None] * self.nb_load_threads  # TODO: threading doesn't really work properly
        self.loaded_xs: List[numpy.ndarray] = [None] * self.nb_load_threads
        self.loaded_ys: List[numpy.ndarray] = [None] * self.nb_load_threads
        self.current_xs: numpy.ndarray = None
        self.current_ys: numpy.ndarray = None

        self.past_last_file: bool = False
        self.past_last_load_file: bool = False
        self.continue_past_last_file: bool = self.wrap_back_after_epoch
        self.continue_past_last_load_file: bool = self.wrap_back_after_epoch or self.reset_when_done

        self.start_all_loading_threads_and_wait_until_current_is_done()
        # We wait so that we can get the shapes as soon as possible. This also loads self.current_xs and
        # self.current_ys.
        self.x_shape: tuple = self.current_xs.shape[1:]
        self.y_shape: tuple = self.current_ys.shape[1:]

    def start_all_loading_threads_and_wait_until_current_is_done(self):
        """Loading thread self.current_file_thread_number will load the file corresponding to
        self.current_file_index. From there on we cycle."""
        self.current_load_file_index = self.current_file_index
        for i in range(self.nb_load_threads):
            thread_nb = (self.current_file_thread_nb + i) % self.nb_load_threads
            self.current_load_file_index = (self.current_file_index + i) % self.get_nb_files()
            self.load_threads[thread_nb] = threading.Thread(
                target=self.load_file, args=(self.file_nbs[self.current_load_file_index], thread_nb))
            self.load_threads[thread_nb].start()

        self.went_to_next_file = True
        file_load_wait_start = time.perf_counter()
        self.load_threads[self.current_file_thread_nb].join()
        self.wait_for_file_to_load_time = time.perf_counter() - file_load_wait_start
        if self.print_debug:
            print("Wait until the first file has finished loading: {} ms".format(
                round(self.wait_for_file_to_load_time * 1000)))

        self.current_xs = self.loaded_xs[self.current_file_thread_nb]
        self.current_ys = self.loaded_ys[self.current_file_thread_nb]

    def load_file(self, file_nb, thread_nb):
        time.sleep(thread_nb * 0.01)  # Will crash without this sleep. (We need to desynchronize the threads.)
        data_loader = self.data_loader_class(self.folder, self.prefix, file_nb)
        if not self.nb_stacked_data:
            self.loaded_xs[thread_nb] = data_loader.load_x()
        else:
            self.loaded_xs[thread_nb] = data_loader.load_stacked_x(self.nb_stacked_data, self.stack_in_last_component)
        self.loaded_ys[thread_nb] = data_loader.load_y()

    def go_to_next_file(self, dont_update_current_in_file_entry: bool = False):
        """The argument dont_update_current_in_file_entry is only set to True in the exceptional case where self.step
        is so large and the current file so small that there is no data to use from this file. Then
        go_to_next_file(dont_update_current_in_file_entry=True) is used to cycle to the next useful file.
        """

        if self.continue_past_last_file:
            self.past_last_file = False  # Reset value.
        if self.continue_past_last_load_file:
            self.past_last_load_file = False

        # Load in the next file (to be loaded), if there still is one.
        self.current_load_file_index += 1
        if self.current_load_file_index >= self.get_nb_files():
            self.past_last_load_file = True
            if self.continue_past_last_load_file:
                self.current_load_file_index = 0

        if not self.past_last_load_file or self.continue_past_last_load_file:
            # Start loading the next file (to load).
            self.load_threads[self.current_file_thread_nb] = \
                threading.Thread(target=self.load_file,
                                 args=(self.file_nbs[self.current_load_file_index], self.current_file_thread_nb))
            self.load_threads[self.current_file_thread_nb].start()

        # Set the current file to the next (already loaded file) (if possible).
        self.current_file_thread_nb = (self.current_file_thread_nb + 1) % self.nb_load_threads
        self.current_file_index += 1
        if self.current_file_index >= self.get_nb_files():
            self.past_last_file = True
            if self.continue_past_last_file:
                self.current_file_index = 0
                self.next_batch_epoch += 1

        if not self.past_last_file or self.continue_past_last_load_file:
            # Update self.current_in_file_entry.
            if not dont_update_current_in_file_entry:
                self.current_in_file_entry = self.current_in_file_entry + self.step - self.current_xs.shape[0]
                # This should be non-negative, as otherwise we did not have to go to the next file.
                # In principle, if the current file is very short and self.step very large, it could happen that
                # self.current_in_file_entry is larger than the next file's length. This will be dealt with after
                # reading from the next file.

            # Read from the new current file (which has already started loading).
            if self.load_threads[self.current_file_thread_nb]:
                self.went_to_next_file = True
                file_load_wait_start = time.perf_counter()
                self.load_threads[self.current_file_thread_nb].join()  # Make sure we are done loading the current file.
                self.wait_for_file_to_load_time = time.perf_counter() - file_load_wait_start
                if self.print_debug:
                    print("{}Wait until file has finished loading: {} ms".format(
                        PRINT_DEBUG_PREFIX,
                        round(self.wait_for_file_to_load_time * 1000)))
            self.current_xs = self.loaded_xs[self.current_file_thread_nb]
            self.current_ys = self.loaded_ys[self.current_file_thread_nb]

            if not dont_update_current_in_file_entry:
                while self.current_in_file_entry >= self.current_xs.shape[0]:
                    # Skip this file
                    self.current_in_file_entry -= self.current_xs.shape[0]
                    # and go to the next. As self.current_in_file_entry is not valid (and has not been updated
                    # in self.generate_batch()) we cannot use the normal update rule.
                    self.go_to_next_file(dont_update_current_in_file_entry=True)

    def generate_batch(self) -> (numpy.ndarray, numpy.numarray):

        while True:
            # Repeat if self.reset_when_done.
            done = False
            while not done:
                # Generate a batch.
                remaining_size = self.batch_size
                batch_xs = numpy.empty((0,) + self.x_shape)  # Empty numpy array of 'length' 0 and the correct shape,
                batch_ys = numpy.empty((0,) + self.y_shape)  # i.e. its shape is (0, <shape of one x/y>).

                self.last_batch_epoch = self.next_batch_epoch
                self.last_batch_file_index = self.current_file_index
                self.batch_generation_start_time = time.perf_counter()
                self.wait_for_file_to_load_time = 0.0
                self.went_to_next_file = False

                while remaining_size > 0:
                    # Fill the batch.
                    file_load_end = self.current_in_file_entry + self.step * (remaining_size - 1) + 1  # exclusive
                    # * (... - 1) + 1: only the first entry in the last step-block needs to be in this file.
                    # Note that we always need that file_load_end = self.current_in_file_entry + 1   mod self.step,
                    # where the + 1 is due to the fact that the entry at file_load_end itself is not included.
                    remainder_completely_in_current_file = file_load_end <= self.current_xs.shape[0]
                    if not remainder_completely_in_current_file:
                        # Load all we can from this file.
                        file_load_end = self.current_xs.shape[0] \
                                        - (self.current_xs.shape[0] - self.current_in_file_entry - 1) % self.step
                        # This is the largest value smaller than self.current_xs.shape[0] which is congruent to
                        # self.current_in_file_entry + 1 modulo self.step.

                    batch_xs = numpy.append(batch_xs, self.current_xs[self.current_in_file_entry:file_load_end:self.step], 0)
                    batch_ys = numpy.append(batch_ys, self.current_ys[self.current_in_file_entry:file_load_end:self.step], 0)
                    self.current_in_file_entry = file_load_end

                    if remainder_completely_in_current_file:
                        remaining_size = 0
                    else:
                        remaining_size = self.batch_size - batch_xs.shape[0]

                        # Go to the next file for the remainder of the batch.
                        self.go_to_next_file()
                        if self.past_last_file:
                            if self.allow_incomplete_last_batch:
                                remaining_size = 0  # Stop filling this batch.
                            done = not self.continue_past_last_file
                            if self.wrap_back_after_epoch:
                                done = done or self.next_batch_epoch > self.nb_epochs  # 'done or' should be superfluous
                            if done:
                                break

                if batch_xs.shape[0] >= self.batch_size or self.allow_incomplete_last_batch:
                    # Only == should ever occur, but just to be safe.
                    self.batch_generation_time = time.perf_counter() - self.batch_generation_start_time
                    if self.print_debug:
                        print("Batch generation time: {} ms".format(
                            round(self.batch_generation_time * 1000.0)))
                    yield batch_xs, batch_ys

            if not self.reset_when_done:
                break

            self.reset(False)  # False as the first files have already again been loaded by self.go_to_next_file().

    def reset(self, load_first_files=True):
        """To allow for later calls, without having to create a new object. Especially useful when
        wrap_back_after_epoch is False and in combination with keras.fit_generator()."""
        self.next_batch_epoch: int = 1

        self.current_file_index = 0
        self.current_in_file_entry = self.offset
        self.current_load_file_index = 0

        if load_first_files:
            self.current_file_thread_nb: int = 0
            self.start_all_loading_threads_and_wait_until_current_is_done()

    def went_to_next_epoch(self) -> bool:
        return self.next_batch_epoch != self.last_batch_epoch

    def get_nb_files(self) -> int:
        return len(self.file_nbs)

    def save(self, save_path: str, save_loaded_data: bool = False):
        """If save_loaded_data is False, self.current_xs etc. will not be pickled. In the loading method they will then
        be recomputed from the data files. (This is less disk and memory intensive and should even by faster because
        compressing these variables will take a while.)"""
        vars_dict = self.set_up_save(save_path, save_loaded_data)
        BatchGenerator.save_vars_dict(save_path, vars_dict)

    def create_save_thread(self, save_path) -> threading.Thread:
        # First make the deep copy of the variables in the main thread.
        vars_dict = self.set_up_save(save_path)
        return threading.Thread(target=BatchGenerator.save_vars_dict, args=(save_path, vars_dict))

    def set_up_save(self, save_path: str, save_loaded_data: bool = False) -> dict:
        folder = os.path.dirname(save_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        # Note: we cannot simply use (e.g.) (compress_)pickle.dump(self, file, compression="lzma"),
        # because of the threading.
        # Save all instance variables except load_threads and the loaded data (as this will temporarily double the
        # (potentially already large) memory requirements.)
        return copy.deepcopy({var: self.__dict__[var] for var in self.__dict__
                              if var != "load_threads"
                              and (save_loaded_data or (var != "current_xs" and var != "current_ys"
                                                        and var != "loaded_xs" and var != "loaded_ys"))})

    @staticmethod
    def save_vars_dict(save_path: str, vars_dict: dict):
        """Used as a second step of save(). Don't use otherwise. Defined explicitly, as it is useful for threading."""
        with open(save_path, "wb") as file:
            compress_pickle.dump(vars_dict, file, compression="lzma")
            # (This requires more memory than looping through self.__dict__ and saving all entries except load_threads
            #  one by one. But in that case we need to be careful that the internal state stays consistent while saving.
            #  In particular, if while saving loaded_xs we move on to the next file, the saved loaded_xs and
            #  loaded_ys will not be compatible.)
            # (For that approach we cannot use compress_pickle, as that does not seem to
            # support dumping multiple objects (without making a list or dict first).)

    @classmethod
    def load_saved_batch_generator(cls, saved_generator_path: str):
        self = cls.__new__(cls)

        with open(saved_generator_path, "rb") as file:
            vars_dict = compress_pickle.load(file, compression="lzma")
        self.__dict__ = vars_dict

        self.load_threads = [None] * self.nb_load_threads

        if "current_xs" not in vars_dict:  # saved with save_loaded_data = False -> load this data from the files
            self.loaded_xs = [None] * self.nb_load_threads
            self.loaded_ys = [None] * self.nb_load_threads
            self.current_xs = None
            self.current_ys = None
            self.start_all_loading_threads_and_wait_until_current_is_done()

        return self

    @staticmethod
    def get_file_nbs_from_first_last(folder: str, first_file_nb, last_file_nb) -> List[int]:
        """
        These values for first_file_nb and last_file_nb are inclusive, and use the syntax for arrays
        when using negative values (e.g. with last_file_nb = -1 we get the last file in the folder).
        """
        total_nb_files = len(os.listdir(folder))
        first_file_nb_mod = first_file_nb % total_nb_files  # Python always returns values in 0..n-1 for % n.
        last_file_nb_mod = last_file_nb % total_nb_files
        return list(range(first_file_nb_mod, last_file_nb_mod + 1))
