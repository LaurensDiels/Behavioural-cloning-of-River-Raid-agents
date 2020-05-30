from typing import Dict, List, Tuple, Union, Callable
import keras
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy
import os
import copy
import threading
import pickle
import compress_pickle
import lz4.frame
import random

from BatchGenerator import BatchGenerator, PRINT_DEBUG_PREFIX
from BatchLearnerSmoothingOptions import BatchLearnerSmoothingOptions
from Time import Time
import SequenceSmoothing


class BatchLearner:
    """Alternative to keras.fit_generator() which also works when we do not know the training data size in advance.
    It uses keras.train_on_batch(), but also prints intermediate outputs (if desired by the user)."""

    EXPONENTIAL_MOVING_AVERAGE_WEIGHT_FACTOR = 0.1
    # We will calculate average times using exponential decaying weights.
    # Lower values will give less fluctuations.

    def __init__(self,
                 fraction_training_validation: int, data_normalization_function: Callable,
                 training_batch_generator: BatchGenerator, validation_batch_generator: BatchGenerator = None,
                 auto_save_folder: str = "./", auto_save_file_name: str = None, save_after_every_n_epochs: int = 1,
                 smoothing_option: BatchLearnerSmoothingOptions
                    = BatchLearnerSmoothingOptions.SIMPLE_MOVING_AVERAGE_EPOCH_WINDOW,
                 print_outputs: bool = True, print_debug: bool = False):
        """
        The argument fraction_training_validation determines the amount of batches that will be designated to
        validation. For example if this is 5, after every 5 training batches, there will follow one validation batch.

        If data_normalization_function is None, we will just accept the batches as they come from the BatchGenerators.
        Otherwise, this function will be called first. It should take in a numpy array of input data (e.g. images)
        and a numpy array of output data (e.g. labels). The output should again be two numpy arrays, one for the
        normalised input (e.g. converted from integer to float), and one for the normalised output
        (e.g. one-hot-encoded).

        We advise to create the BatchGenerators using create_batch_generator_for_batch_learner. If
        validation_batch_generator is not explicitly set, we also use the training_batch_generator to produce
        validation batches.

        If auto_save_folder and auto_save_file_name are set to a valid path, we will save the learner (including the
        model and the learning history) after every save_after_every_n_epochs epochs to this file.
        (This is especially useful in case the program is interrupted during training.) In other words, in this case
        we will automatically save checkpoints.
        Importantly auto_save_file_name should not have an extension.
        The learner can be reloaded using BatchLearner.load_saved_batch_learner.

        The parameter smoothing_option allows the batch metrics (e.g. loss) to be smoothed. This is similar to the
        option reset_metrics in keras.train_on_batch(). To get the smoothed there with reset_metrics = false,
        use BatchLearnerSmoothingOptions.RUNNING_AVERAGE_RESET_AFTER_EPOCH. By default we use
        BatchLearnerSmoothingOptions.SIMPLE_MOVING_AVERAGE_EPOCH_WINDOW.

        If print_outputs is true, the current epoch number, batch number (both global and relative to epoch),
        training (or validation) metrics and ETAs, both for this epoch as globally, will be printed.
        """

        self.training_batch_generator: BatchGenerator = training_batch_generator
        self.validation_batch_generator: BatchGenerator = validation_batch_generator
        if not self.validation_batch_generator:
            self.validation_batch_generator = training_batch_generator

        self.data_normalization_function: Callable = data_normalization_function

        self.current_batch_x: numpy.ndarray = None
        self.current_batch_y: numpy.ndarray = None
        self.current_batch_is_training: bool = True
        self.next_batch_x: numpy.ndarray = None  # The generator might get another batch while the current
        self.next_batch_y: numpy.ndarray = None  # batch is still being processed.
        self.next_batch_is_training: bool = True

        self.print_outputs: bool = print_outputs
        self.smoothing_option = smoothing_option
        self.reset_train_history_smoothing_now = False
        self.reset_validation_history_smoothing_now = False
        self.print_debug = print_debug
        self.fraction_training_validation: int = fraction_training_validation
        self.batches_since_last_validation: int = 0  # Start first with training batches.

        self.is_reloaded = False
        self.global_start_time: float = 0.0
        self.epoch_start_time: float = 0.0
        self.batch_start_time: float = 0.0
        self.current_batch_time: float = 0.0
        self.average_batch_time: float = 0.0
        self.current_batch_model_process_time: float = 0.0
        self.save_wait_time: float = 0.0

        self.current_batch_generation_time: float = 0.0
        self.current_batch_went_to_next_file: bool = False
        self.current_batch_wait_for_file_time: float = 0.0
        self.current_batch_wait_for_generator_time: float = 0.0

        self.current_batch_normalization_time: float = 0.0  # Normalization using data_normalization_function,
        self.next_batch_normalization_time: float = 0.0     # nothing to do with BatchNorm

        # Epochs are exclusively determined by the training generator.
        self.last_batch_epoch: int = -1  # Must start lower than epochs (which start at 1).
        self.current_batch_epoch: int = 0
        self.next_batch_epoch: int = 0
        self.last_batch_training_file_index: int = -1
        self.current_batch_training_file_index: int = -1
        self.current_batch_training_file_length: int = -1
        self.global_batch_nb: int = 0
        self.in_epoch_batch_nb: int = 0
        self.epoch_first_batch_nb: int = 0
        self.in_epoch_training_batch_nb: int = 0
        self.in_file_training_batch_nb: int = 0

        self.estimated_training_batches_per_file: float = 0.0  # average
        # The number of batches in an epoch might differ per epoch, because epochs are based solely on training data.
        self.estimated_training_batches_per_epoch: int = 0
        self.smoothed_global_ETA: float = 0.0
        self.smoothed_epoch_ETA: float = 0.0

        self.model: keras.Model = None

        # Possibly smoothed
        self.train_history: Dict[str, List[float]] = {}
        self.validation_history: Dict[str, List[float]] = {}

        # The window sizes are (might be) relevant for smoothing, but will only be fixed (and used) after the
        # first epoch. During the first epoch they will continually be updated to become ever larger, see
        # self.update_other_variables for more details.
        self.window_sizes_fixed: bool = False
        self.validation_window_size: int = self.global_batch_nb
        self.training_window_size: int = self.global_batch_nb
        self.epoch_window_size: int = self.global_batch_nb

        # Not smoothed: contains as keys the metric_names with unsmoothed values.
        # Also has keys "is_training" (True / False for validation) "batch_nb" and "time".
        # (In principle "is_training" and "epoch_nb" can also just be found using self.fraction_training_validation and
        # self.estimated_training_batches_per_epoch.)
        self.raw_history: Dict[str, Union[List[float], List[bool], List[int]]] = {}

        if auto_save_file_name:  # It's okay if auto_save_folder is empty (or even None).
            self.auto_save_path_without_extension = os.path.join(auto_save_folder, auto_save_file_name)
            if auto_save_folder and not os.path.exists(auto_save_folder):
                os.makedirs(auto_save_folder)
        else:
            self.auto_save_path_without_extension = None
        self.save_after_every_n_epochs = save_after_every_n_epochs

        self.should_save: bool = False
        self.main_save_thread: threading.Thread = None

    def start_training_model(self, model: keras.Model):
        """The Keras model should have been compiled."""
        self.model = model
        self.train_history = {metric_name: [] for metric_name in self.model.metrics_names}
        self.validation_history = {metric_name: [] for metric_name in self.model.metrics_names}
        self.raw_history = {metric_name: [] for metric_name in self.model.metrics_names}
        self.raw_history["is_training"] = []
        self.raw_history["epoch_nb"] = []
        self.raw_history["time"] = []

        self.global_start_time = time.perf_counter()
        self.epoch_start_time = time.perf_counter()
        self.batch_start_time = time.perf_counter()

        self.train()

    def train(self):
        get_next_batch_thread = threading.Thread(target=self.get_next_batch)
        get_next_batch_thread.start()
        # This thread will update self.next_batch_x, _y, self.next_batch_is_training and self.no_more_training_batches.

        while True:  # until we break after finding out we have finished our number of epochs

            self.should_save = False
            current_batch_generation_wait_start = time.perf_counter()
            get_next_batch_thread.join()
            self.current_batch_wait_for_generator_time = time.perf_counter() - current_batch_generation_wait_start

            self.current_batch_x = self.next_batch_x
            self.current_batch_y = self.next_batch_y
            self.current_batch_is_training = self.next_batch_is_training
            self.current_batch_normalization_time = self.next_batch_normalization_time
            self.update_variables_for_or_from_batch_generators()
            # To avoid getting already new variables when querying the updating batch generators.

            if self.next_batch_epoch > self.get_nb_epochs():
                break

            self.should_save = self.current_batch_is_training and self.auto_save_path_without_extension and \
                            self.went_to_next_epoch() and \
                            self.next_batch_epoch % self.save_after_every_n_epochs == 1 % self.save_after_every_n_epochs
                          # 1 (instead of 0) as we start counting from 1; second % n to accommodate the case where n = 1
            # Only in training batches, as this is where self.went_to_next_epoch() gets updated.

            save_training_batch_generator_thread = None
            save_validation_batch_generator_thread = None
            if self.should_save:
                # To know this we only need to know the next batch epoch, which we have updated above.
                save_training_batch_generator_thread = \
                    self.training_batch_generator.create_save_thread(self.auto_save_path_without_extension + ".newtbg")
                save_validation_batch_generator_thread = \
                    self.validation_batch_generator.create_save_thread(self.auto_save_path_without_extension + ".newvbg")

            # Already start loading the next batch
            get_next_batch_thread = threading.Thread(target=self.get_next_batch)
            get_next_batch_thread.start()

            current_batch_model_process_start_time = time.perf_counter()
            batch_metrics = self.model.train_on_batch(self.current_batch_x, self.current_batch_y, reset_metrics=True) \
                if self.current_batch_is_training \
                else self.model.test_on_batch(self.current_batch_x, self.current_batch_y, reset_metrics=True)
            # True as the metrics history is shared between train and test in the model and we will do smoothing
            # ourselves.
            self.current_batch_model_process_time = time.perf_counter() - current_batch_model_process_start_time

            self.update_other_variables(batch_metrics)

            if self.print_outputs:
                self.print_output()

            if self.should_save:
                self.save(save_training_batch_generator_thread=save_training_batch_generator_thread,
                          save_validation_batch_generator_thread=save_validation_batch_generator_thread)

            if self.print_debug:
                self.print_debug_metrics()

            if self.should_save:
                self.batch_start_time = time.perf_counter()
            # If we needed to wait for the previous save to finish, this time would otherwise be considered as part
            # of the time needed to process the next batch.

    def continue_training_model(self):
        """Only use after loading with load_saved_batch_learner."""
        if self.is_reloaded:
            self.batch_start_time = time.perf_counter()
            self.train()

    def get_next_batch(self):
        self.next_batch_is_training = (self.batches_since_last_validation % (self.fraction_training_validation + 1)
                                        != self.fraction_training_validation)
        # + 1: e.g. when the fraction is 4, one in every 5 batches should be for validation
        # (and the other 4 (mod 5: 0, 1, 2, 3) for training).

        unnormalized_batch_x = None
        unnormalized_batch_y = None
        if self.next_batch_is_training:
            if self.training_batch_generator.next_batch_epoch <= self.get_nb_epochs():
                unnormalized_batch_x, unnormalized_batch_y = next(self.training_batch_generator.generate_batch())
        else:
            unnormalized_batch_x, unnormalized_batch_y = next(self.validation_batch_generator.generate_batch())

        if unnormalized_batch_x is not None:
            normalization_start_time = time.perf_counter()
            self.next_batch_x, self.next_batch_y = self.data_normalization_function(unnormalized_batch_x,
                                                                                    unnormalized_batch_y)
            self.next_batch_normalization_time = time.perf_counter() - normalization_start_time

    def update_variables_for_or_from_batch_generators(self):
        """Here we update all variables that must be changed before the next batch is generator, either because the
        batch generation process itself needs it, or because these variables are only valid in the current state
        of the generators and will change when generating the next batch."""
        self.current_batch_epoch = self.training_batch_generator.last_batch_epoch
        self.current_batch_training_file_index = self.training_batch_generator.last_batch_file_index
        self.current_batch_training_file_length = self.training_batch_generator.current_xs.shape[0]
        self.next_batch_epoch = self.training_batch_generator.next_batch_epoch

        if self.current_batch_is_training:
            self.batches_since_last_validation += 1
        else:
            self.batches_since_last_validation = 0

        bg = self.training_batch_generator if self.current_batch_is_training else self.validation_batch_generator
        self.current_batch_generation_time = bg.batch_generation_time
        self.current_batch_went_to_next_file = bg.went_to_next_file
        self.current_batch_wait_for_file_time = bg.wait_for_file_to_load_time

    def update_other_variables(self, batch_metrics: List[float]):
        self.global_batch_nb += 1

        # The current batch is the last produced batch for the training generator.
        is_first_epoch = self.current_batch_epoch == 1
        is_new_epoch = self.current_batch_epoch > self.last_batch_epoch

        if is_new_epoch:
            if self.current_batch_is_training:
                self.estimated_training_batches_per_epoch = self.in_epoch_training_batch_nb
                self.in_epoch_training_batch_nb = 1
            self.in_epoch_batch_nb = 1
            self.epoch_first_batch_nb = self.global_batch_nb

            if self.smoothing_option == BatchLearnerSmoothingOptions.RUNNING_AVERAGE_RESET_AFTER_EPOCH:
                self.reset_train_history_smoothing_now = True
                self.reset_validation_history_smoothing_now = True
        else:
            self.in_epoch_batch_nb += 1
            if self.current_batch_is_training:
                self.in_epoch_training_batch_nb += 1
        self.last_batch_epoch = self.current_batch_epoch

        is_new_training_file = self.current_batch_training_file_index \
                                != self.last_batch_training_file_index
        # Note that we can only change training file when training is True.)
        if is_new_training_file:
            if is_first_epoch:
                # Update self.estimated_training_batches_per_file.
                nb_processed_files = self.current_batch_training_file_index + 1
                # + 1: As we will include an estimate for the number of batches in the current file.

                # Update running average.
                estimated_training_batches_in_this_file = math.ceil(
                                                self.current_batch_training_file_length
                                                / (self.get_training_batch_size() * self.get_training_data_step()))
                if nb_processed_files == 1:
                    self.estimated_training_batches_per_file = estimated_training_batches_in_this_file
                else:
                    self.estimated_training_batches_per_file = estimated_training_batches_in_this_file \
                                                               / nb_processed_files \
                                                               + self.estimated_training_batches_per_file \
                                                                    * (nb_processed_files - 1) / nb_processed_files
            self.in_file_training_batch_nb = 1
        else:
            self.in_file_training_batch_nb += 1
        self.last_batch_training_file_index = self.current_batch_training_file_index

        if is_first_epoch:
            self.estimated_training_batches_per_epoch = self.estimated_training_batches_per_file \
                                                        * (self.get_nb_training_files())
            # After the first_epoch we have seen all batches and hence know their amount.

        self.raw_history["is_training"].append(self.current_batch_is_training)
        self.raw_history["epoch_nb"].append(self.current_batch_epoch)
        self.raw_history["time"].append(time.perf_counter())

        if is_first_epoch:
            self.window_sizes_fixed = False
            self.validation_window_size = self.global_batch_nb
            self.training_window_size = self.global_batch_nb
            self.epoch_window_size = self.global_batch_nb
            # The concrete value does not matter. What is important is that these window sizes are larger than the
            # current history lengths (<= self.global_batch_nb - 1). This ensures we don't reset the smoothing
            # in the case of resetting moving average or switch from running average to simple moving average
            # in the first epoch itself, but only in the beginning (up to a few batches) of the second.
            # Concretely, this way in the code below we always have len(history) < window_size.
        else:
            self.window_sizes_fixed = True
            self.validation_window_size = math.ceil(self.estimate_average_validation_batches_per_epoch())
            self.training_window_size = self.fraction_training_validation * self.validation_window_size
            self.epoch_window_size = self.validation_window_size + self.training_window_size
            # It's important that these value do not change over the epochs. Therefore we don't use
            # validation_window_size = self.estimate_validation_batches_in_this_epoch().
            # Furthermore it's important that epoch_window_size is a multiple of self.fraction_training_validation + 1.
            # Therefore we also don't use training_window_size = self.estimate_training_batches_per_epoch().
            # (Alternatively we could (now) also use the "is_training" and "epoch_nb" information in self.raw_history.)
            # Finally, we also require that these are overestimations of the actual number of batches in an epoch, so
            # that we do not find at the beginning of the second epoch that we should already have reset/switched to
            # a simple moving average near the end of the first epoch. Hence the ceil.

        for i in range(len(batch_metrics)):
            if self.current_batch_is_training:
                history = self.train_history[self.model.metrics_names[i]]
                should_reset_flag = self.reset_train_history_smoothing_now
                window_size = self.training_window_size
            else:
                history = self.validation_history[self.model.metrics_names[i]]
                should_reset_flag = self.reset_validation_history_smoothing_now
                window_size = self.validation_window_size

            new = None  # To get rid of 'might be unassigned' warning
            if self.smoothing_option == BatchLearnerSmoothingOptions.NO_SMOOTHING or not history or should_reset_flag:
                new = batch_metrics[i]
                if self.current_batch_is_training:
                    self.reset_train_history_smoothing_now = False
                else:
                    self.reset_validation_history_smoothing_now = False
            else:
                old_avg = history[-1]
                if self.smoothing_option == BatchLearnerSmoothingOptions.RUNNING_AVERAGE:
                    old_size = len(history)
                    new = (old_avg * old_size + batch_metrics[i]) / (old_size + 1)
                elif self.smoothing_option == BatchLearnerSmoothingOptions.RUNNING_AVERAGE_RESET_AFTER_EPOCH:
                    old_size = len(history) % window_size
                    new = (old_avg * old_size + batch_metrics[i]) / (old_size + 1)
                elif self.smoothing_option == BatchLearnerSmoothingOptions.SIMPLE_MOVING_AVERAGE_EPOCH_WINDOW:
                    if len(history) < window_size:  # simple running average (~ in first epoch)
                        old_size = len(history)
                        new = (old_avg * old_size + batch_metrics[i]) / (old_size + 1)
                    else:  # simple moving average
                        first_in_old_avg = \
                            self.raw_history[self.model.metrics_names[i]][-self.epoch_window_size]
                        # This is the unsmoothed version of history[-window_size].
                        new = old_avg + (batch_metrics[i] - first_in_old_avg) / window_size

            history.append(new)
            self.raw_history[self.model.metrics_names[i]].append(batch_metrics[i])

        self.current_batch_time = time.perf_counter() - self.batch_start_time
        if self.global_batch_nb == 1:
            self.average_batch_time = self.current_batch_time
            self.smoothed_global_ETA = self.estimate_remaining_total_time()
            self.smoothed_epoch_ETA = self.estimate_remaining_epoch_time()
        else:
            self.average_batch_time = (BatchLearner.EXPONENTIAL_MOVING_AVERAGE_WEIGHT_FACTOR
                                        * self.current_batch_time) \
                                      + ((1 - BatchLearner.EXPONENTIAL_MOVING_AVERAGE_WEIGHT_FACTOR)
                                        * self.average_batch_time)
            # Running average with exponentially decaying weights
            self.smoothed_global_ETA = (BatchLearner.EXPONENTIAL_MOVING_AVERAGE_WEIGHT_FACTOR
                                            * self.estimate_remaining_total_time()) \
                                        + ((1 - BatchLearner.EXPONENTIAL_MOVING_AVERAGE_WEIGHT_FACTOR)
                                            * (self.smoothed_global_ETA - self.current_batch_time))
            if is_new_epoch:
                self.smoothed_epoch_ETA = self.estimate_remaining_epoch_time()
            else:
                self.smoothed_epoch_ETA = (BatchLearner.EXPONENTIAL_MOVING_AVERAGE_WEIGHT_FACTOR
                                           * self.estimate_remaining_epoch_time()) \
                                         + ((1 - BatchLearner.EXPONENTIAL_MOVING_AVERAGE_WEIGHT_FACTOR)
                                            * (self.smoothed_epoch_ETA - self.current_batch_time))
            # - current_batch_time as this time has expired since the last estimate
        if self.smoothed_epoch_ETA < 0:
            self.smoothed_epoch_ETA = 0
        if self.smoothed_global_ETA < 0:
            self.smoothed_global_ETA = 0

        self.batch_start_time = time.perf_counter()
        if is_new_epoch:
            self.epoch_start_time = time.perf_counter() - self.current_batch_time

    def print_output(self):
        """Example:
        Epoch 7/10 - Batch 8,773‬/12,640 - global ETA: 1h 4m 27s
        |	Batch 1,189/1,264 - epoch ETA: 1m 15s
        |	|	Loss: 1.308 - Accuracy: 0.736   (Training)
        ----------------------------------------------

        During the first epoch the ETAs might fluctuate as we do not yet know how many batches there are in an epoch.
        """

        print("Epoch: {}/{} - Batch {:,}/{}{:,} - global ETA: {}".format(
            self.get_last_batch_epoch_nb(),
            self.get_nb_epochs(),
            self.global_batch_nb,
            "" if self.batches_estimations_are_nonchanging() else "~",  # ~ if we are not sure yet
            math.ceil(self.estimate_total_batches()),  # ceil as it's better to be pessimistic
            Time(math.ceil(self.smoothed_global_ETA)).to_string()
        ))

        print("|\tBatch {:,}/{}{:,} - epoch ETA: {}".format(
            self.in_epoch_batch_nb,
            "" if self.batches_estimations_are_nonchanging() else "~",
            math.ceil(self.estimate_batches_in_this_epoch()),
            Time(math.ceil(self.smoothed_epoch_ETA)).to_string()
        ))

        history = self.train_history if self.current_batch_is_training else self.validation_history
        training_or_validation_string = "\t(Training)" if self.current_batch_is_training else "\t(Validation)"
        print("|\t|\t", end="")
        for i in range(len(self.model.metrics_names)):
            metrics_name = self.model.metrics_names[i]
            print("{0:s}: {1:.3f}".format(metrics_name, history[metrics_name][-1]),
                  end=" - " if i != len(self.model.metrics_names) - 1 else training_or_validation_string)

        print("\n----------------------------------------------")

    def print_debug_metrics(self):
        bgt = self.current_batch_generation_time
        if bgt <= 0.0:  # == 0.0 can happen for very small batch sizes
            bgt = 0.0001

        print("Complete batch time: {} ms".format(round(self.current_batch_time * 1000.0)))
        print("{}Wait for batch generation time: {} ms ({}%)".format(
            PRINT_DEBUG_PREFIX,
            round(self.current_batch_wait_for_generator_time * 1000.0),
            round(self.current_batch_wait_for_generator_time / self.current_batch_time * 100.0)
        ))
        print("{}Batch loading time: {} ms".format(
            PRINT_DEBUG_PREFIX * 2, round(bgt * 1000.0)))
        if self.current_batch_went_to_next_file:
            print("{}Wait for file to finish loading time: {} ms ({}%)".format(
                PRINT_DEBUG_PREFIX * 3,
                round(self.current_batch_wait_for_file_time * 1000.0),
                round((self.current_batch_wait_for_file_time / bgt * 100.0))
            ))
        print("{}Batch normalization time: {} ms".format(
            PRINT_DEBUG_PREFIX * 2, round(self.current_batch_normalization_time * 1000.0)))
        print("{}Batch model processing time: {} ms ({}%)".format(
            PRINT_DEBUG_PREFIX,
            round(self.current_batch_model_process_time * 1000.0),
            round(self.current_batch_model_process_time / self.current_batch_time * 100.0)
        ))
        if self.should_save:
            print("+++++++++++++++++++++++++++++++++++++++++++++++")  # not part of the previous batch
            print("Wait until previous save has finished saving time: {} ms".format(
                round(self.save_wait_time * 1000)
            ))
        print("+++++++++++++++++++++++++++++++++++++++++++++++")

    def make_history_plot(self, path_to_file: str = None, title: str = "",
                          use_smoothing: bool = True,
                          xlim: Dict[str, Tuple[int, int]] = None, ylim: Dict[str, Tuple[int, int]] = None,
                          force_x_tick_1: bool = False):
        """
        If path_to_file is left unspecified (None) we don't save the plot.

        When use_smoothing set to True, we will use the BatchLearnerSmoothingOptions value set at initialization.
        Otherwise we use the raw unsmoothed data.

        xlim and ylim set the axis boundaries. E.g. ylim={"accuracy": (0, 1), "loss": (0, 5)}

        With force_x_tick_1 set to True, we will place a tick per epoch. When set to False, matplotlib decides
        how to put the ticks.
        """

        plot_history_dict = self.get_history(use_smoothing)

        BatchLearner.make_history_plot_from_plot_history_dict(plot_history_dict,
                                                              path_to_file, title,
                                                              xlim, ylim,
                                                              force_x_tick_1)

    @staticmethod
    def make_history_plot_from_plot_history_dict(plot_history_dict: Dict[str, Dict[str, List[Tuple[float, float]]]],
                                                 path_to_file: str = None, title: str = "",
                                                 xlim: Dict[str, Tuple[int, int]] = None,
                                                 ylim: Dict[str, Tuple[int, int]] = None,
                                                 force_x_tick_1: bool = False
                                                 ):
        """
        (Normally you should use make_history_plot. The use-case of this method is when you want to do more complicated
        post-smoothing, or use another raw_history than the instance variable. In this cases use get_history_from_raw
        to obtain plot_history_dict.)

        plot_history_dict is a dictionary of the type obtained from get_history or get_history_from_raw. The other
        parameters are the same as in make_history_plot."""

        metric_names = list(plot_history_dict["training"].keys())
        nb_metrics = len(metric_names)
        fig, axes = plt.subplots(nb_metrics, sharex=True)
        fig.suptitle(title)
        axes[-1].set_xlabel("Epoch")  # Since we share the x-axis, we only need to label the bottom subplot.

        for i in range(nb_metrics):
            metric_name = metric_names[i]

            for t_or_v in ["training", "validation"]:
                x = [plot_history_dict[t_or_v][metric_name][i][0]
                     for i in range(len(plot_history_dict[t_or_v][metric_name]))]
                y = [plot_history_dict[t_or_v][metric_name][i][1]
                     for i in range(len(plot_history_dict[t_or_v][metric_name]))]
                axes[i].plot(x, y)

            axes[i].set_ylabel(metric_name.capitalize())
            if xlim and metric_name in xlim:
                axes[i].set_xlim(xlim[metric_name])
            if ylim and metric_name in ylim:
                axes[i].set_ylim(ylim[metric_name])

            if force_x_tick_1:
                axes[i].xaxis.set_major_locator(plticker.MultipleLocator(base=1))  # Set ticks at 0, 1, 2 (epochs), etc.

        axes[-1].legend(['Training', 'Validation'], loc="lower right")
        # Since this is the same everywhere, we only add the legend to the bottom subplot.
        if path_to_file:
            folder = os.path.dirname(path_to_file)
            if folder:  # If the path is relative, this might be empty.
                if not os.path.exists(folder):
                    os.makedirs(folder)
            fig.savefig(path_to_file)

    def get_history(self, use_smoothing: bool = True,
                    force_raw_basis: bool = False,
                    post_smoothing_function: Callable = None,
                    **post_smoothing_function_kwargs) \
            -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
        """
        Returns a dictionary with keys "training" and "validation" and values the corresponding history
        dictionaries with as keys the quality metrics (such as "loss" and "accuracy") and as values lists of tuples
        (epoch_nb_float, quality), where epoch_nb_float is a float between 0 and the total number of epochs: e.g.
        0.5 means that this value comes from a batch in the middle of the first (zeroth) epoch.

        With use_smoothing set to True, we will use the same smoothing method supplied at initialization. Otherwise
        we do not smooth anything.

        The other two parameters are advanced options and should under normal circumstances be left at their default
        values. With force_raw_basis set to True we will ignore the (likely smoothed, depending on the chosen
        BatchLearnerSmoothingOption) self.train_history and self.validation_history dictionaries, and do the smoothing
        afterwards using post_smoothing_function (if it is set to something other than None, and if use_smoothing
        is True). Relevant values for post_smoothing_function can be found in SequenceSmoothing.py. Possible parameters
        to post_smoothing_function can be passed using keywords (**post_smoothing_function_kwargs). Since these might be
        different between training and validation, they must be passed with a prefix t_ for training and v_ for
        validation. The prefixes will be removed before calling post_smoothing_function for the relevant history
        (training or validation). E.g.

        t_window_size=self.training_window_size

        and

        v_window_size=self.validation_window_size

        when post_smoothing_function = SequenceSmoothing.simple_moving_average_backwards. Note that in contrast to
        the BatchLearnerSmoothingOptions which must be calculable online, also offline smoothing (incorporating
        future values) can be used.
        """

        metric_names = list(self.train_history.keys())
        # Safer than self.model.metrics_names, as this also works when this batch learner was loaded without a model.

        if force_raw_basis:
            psf = None
            psf_kwargs = {}
            if use_smoothing:
                if post_smoothing_function:
                    psf = post_smoothing_function
                    psf_kwargs = post_smoothing_function_kwargs
                else:
                    if self.smoothing_option is BatchLearnerSmoothingOptions.NO_SMOOTHING:
                        pass  # keep psf = None and psf_kwargs = None
                    elif self.smoothing_option is BatchLearnerSmoothingOptions.RUNNING_AVERAGE:
                        psf = SequenceSmoothing.running_average
                    elif self.smoothing_option is BatchLearnerSmoothingOptions.RUNNING_AVERAGE_RESET_AFTER_EPOCH:
                        psf = SequenceSmoothing.resetting_running_average_irregular_split
                        est_avg_val_batches = self.estimate_average_validation_batches_per_epoch()
                        est_val_batches_in_epochs = [est_avg_val_batches]  # TODO: not exact, also not the same as in self.update_other_variables, but better?
                        for i in range(1, self.get_nb_epochs()):
                            est_val_batches_in_epochs.append(round(i * est_avg_val_batches)
                                                             - est_val_batches_in_epochs[-1])
                            # In this way we don't accumulate rounding errors:
                            # \sum_{k \leq i} est_val_batches_in_epochs[i] is at most 1 different from
                            # i * est_avg_val_batches.

                        psf_kwargs = {"t_reset_after_nbs":
                                          [self.estimate_training_batches_per_epoch()] * self.get_nb_epochs(),
                                      "v_reset_after_nbs":
                                          est_val_batches_in_epochs}
                    elif self.smoothing_option is BatchLearnerSmoothingOptions.SIMPLE_MOVING_AVERAGE_EPOCH_WINDOW:
                        psf = SequenceSmoothing.simple_moving_average_backwards
                        psf_kwargs = {"t_window_size": self.training_window_size,
                                      "v_window_size": self.validation_window_size}
                        # Same settings as in self.update_other_variables

            return BatchLearner.get_history_from_raw(self.raw_history,
                                                     self.estimate_average_batches_per_epoch(),
                                                     self.fraction_training_validation,
                                                     metric_names,
                                                     psf,
                                                     **psf_kwargs)

        # From now onwards force_raw_basis == False
        history = {t_or_v: {metric_name: [] for metric_name in metric_names} for t_or_v in ["training", "validation"]}

        batches_since_last_validation = 0
        training_batch_nb = -1  # so that after the += 1 we start at 0
        validation_batch_nb = -1
        # (Instead of using these counters, we could also use
        #   training = self.is_training_batch_nb(i)
        #   training_batch_nb = self.global_batch_nb_to_training_batch_nb(i)
        #   validation_batch_nb = self.global_batch_nb_to_validation_batch_nb(i)
        #  in the for loop.)
        # (As we added keys "is_training" and "epoch_nb", we could also use this information instead.)

        for i in range(len(self.raw_history[metric_names[0]])):
            epoch_nb_float = i / self.estimate_average_batches_per_epoch()
            if batches_since_last_validation == self.fraction_training_validation:
                training = False
                batches_since_last_validation = 0
                validation_batch_nb += 1
            else:
                training = True
                batches_since_last_validation += 1
                training_batch_nb += 1
            t_or_v = "training" if training else "validation"
            for metric_name in metric_names:
                if not use_smoothing:
                    metric_value = self.raw_history[metric_name][i]
                else:
                    if training:
                        metric_value = self.train_history[metric_name][training_batch_nb]
                    else:
                        metric_value = self.validation_history[metric_name][validation_batch_nb]
                history[t_or_v][metric_name].append((epoch_nb_float, metric_value))

        return history

    @staticmethod
    def get_history_from_raw(raw_history: Dict[str, List[float]],
                             average_batches_per_epoch: float,
                             fraction_training_validation: int,
                             metric_names: List[str] = None,
                             post_smoothing_function: Callable = None,
                             **post_smoothing_function_kwargs) \
            -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
        """
        (There are only few situations where this method would be used instead of get_history_from_raw.)

        See get_history in the case of force_raw_basis=True for the return value and meaning and use of
        post_smoothing_function and **post_smoothing_function_kwargs.

        We now explain the other parameters. raw_history is a dictionary in the style of the instance variable
        self.raw_history: it should have string keys and as values lists of floats, at least for those keys in
        metric_names. If metric_names=None, all keys of raw_history will be used (which in the case of the instance
        variable self.raw_history also contains the batch time and (these days) training/validation and epoch number
        indicators).

        The parameter average_batches_per_epoch is used to calculate the epoch_nb_float (again, see get_history): the
        i-th batch will get i / average_batches_per_epoch as epoch float.
        Using fraction_training_validation we can distinguish between training and validation batches: the first
        fraction_training_validation batches are training, the next is validation. This then repeats until we are out of
        batches.
        """

        if metric_names is None:
            metric_names = list(raw_history.keys())

        history = {t_or_v: {metric_name: [] for metric_name in metric_names} for t_or_v in ["training", "validation"]}

        # Training and validation indices (batch numbers)
        indices = {"training": [i for i in range(len(raw_history[metric_names[0]]))
                                if i % (fraction_training_validation + 1) != fraction_training_validation],
                   "validation": [i for i in range(len(raw_history[metric_names[0]]))
                                  if i % (fraction_training_validation + 1) == fraction_training_validation]}

        for t_or_v in ["training", "validation"]:
            t_or_v_prefix = t_or_v[0] + "_"

            t_or_v_post_smoothing_function_kwargs = \
                {key[2:]: value for key, value in post_smoothing_function_kwargs.items()
                 if key.startswith(t_or_v_prefix)}

            for metric_name in metric_names:
                metric_values = [raw_history[metric_name][i] for i in indices[t_or_v]]
                if post_smoothing_function is not None:
                    metric_values = post_smoothing_function(metric_values, **t_or_v_post_smoothing_function_kwargs)

                for j in range(len(metric_values)):
                    history[t_or_v][metric_name].append(
                        (indices[t_or_v][j] / average_batches_per_epoch,  # epoch_nb_float
                         metric_values[j])
                    )

        return history

    def get_input_shape(self) -> numpy.shape:
        return self.training_batch_generator.x_shape

    def get_nb_epochs(self) -> int:
        return self.training_batch_generator.nb_epochs

    def get_training_batch_size(self) -> int:
        return self.training_batch_generator.batch_size

    def get_training_data_step(self) -> int:
        return self.training_batch_generator.step

    def get_nb_training_files(self) -> int:
        return self.training_batch_generator.get_nb_files()

    def get_total_runtime(self) -> float:
        return self.raw_history["time"][-1] - self.global_start_time

    def get_last_batch_epoch_nb(self) -> int:
        return self.training_batch_generator.last_batch_epoch

    def went_to_next_epoch(self) -> bool:
        return self.training_batch_generator.went_to_next_epoch()

    def estimate_training_batches_per_epoch(self) -> int:
        return self.estimated_training_batches_per_epoch

    def estimate_average_validation_batches_per_epoch(self) -> float:
        return self.estimate_training_batches_per_epoch() / self.fraction_training_validation

    def estimate_validation_batches_in_this_epoch(self) -> int:
        # TODO
        return round(self.estimate_average_validation_batches_per_epoch())

    def estimate_average_batches_per_epoch(self) -> float:
        return self.estimate_training_batches_per_epoch() + self.estimate_average_validation_batches_per_epoch()

    def estimate_batches_in_this_epoch(self) -> float:
        return self.estimate_training_batches_per_epoch() + self.estimate_validation_batches_in_this_epoch()

    def estimate_total_batches(self) -> float:
        return self.estimate_average_batches_per_epoch() * self.get_nb_epochs()

    def estimate_total_training_batches(self) -> int:
        return self.get_nb_epochs() * self.estimate_training_batches_per_epoch()

    def estimate_total_validation_batches(self) -> int:
        return round(self.get_nb_epochs() * self.estimate_average_validation_batches_per_epoch())

    def batches_estimations_are_nonchanging(self) -> bool:
        return self.training_batch_generator.last_batch_epoch >= 2

    def estimate_epoch_time(self) -> float:
        return self.average_batch_time * self.estimate_average_batches_per_epoch()

    def estimate_total_time(self) -> float:
        return self.estimate_epoch_time() * self.get_nb_epochs()

    def estimate_remaining_epoch_time(self) -> float:
        return (self.estimate_average_batches_per_epoch() - self.in_epoch_batch_nb) * self.average_batch_time

    def estimate_remaining_total_time(self) -> float:
        return (self.estimate_total_batches() - self.global_batch_nb) * self.average_batch_time

    def is_training_batch_nb(self, nb: int) -> bool:
        # nb starts counting from 0
        return not self.is_validation_batch_nb(nb)

    def is_validation_batch_nb(self, nb: int) -> bool:
        return (nb + 1) % (self.fraction_training_validation + 1) == 0

    def global_batch_nb_to_validation_batch_nb(self, nb: int) -> int:
        """If nb does not refer to a validation batch, then we return the next validation batch number."""
        return nb // (self.fraction_training_validation + 1)
        # Note: for a validation batch this is the same as (nb + 1) // (self.fraction_training_validation + 1) - 1
        # where the -1 is to start counting at 0.

    def global_batch_nb_to_training_batch_nb(self, nb: int) -> int:
        """If nb does not refer to a training batch, then we return the next training batch number."""
        return nb - self.global_batch_nb_to_validation_batch_nb(nb)

    def save(self, save_path_without_extension: str = None, save_model: bool = True, save_batch_generators: bool = True,
             save_training_batch_generator_thread: threading.Thread = None,
             save_validation_batch_generator_thread: threading.Thread = None,
             save_rest_of_learner_thread: threading.Thread = None):
        """
        Load the resulting files using load_batch_learner.
        If save_path is None we use the auto save path (provided at initialisation),
        and if it is supplied explicitly, save_path should not have an extension.

        If no threads are specified (as is the case by default), these will be created automatically. These should
        only be manually specified when creating them at this time would be inadvisable (e.g. because we want to
        save a slightly out of date generator instead of the one in the current state).
        """
        if save_path_without_extension:
            # Make the path folder, if it does not yet exist.
            folder = os.path.dirname(save_path_without_extension)
            if folder:
                if not os.path.exists(folder):
                    os.makedirs(folder)
        else:
            save_path_without_extension = self.auto_save_path_without_extension
            # The corresponding should already exist, unless we did not supply an auto save path,
            # in which case we now throw an exception.
            if not save_path_without_extension:
                raise NotADirectoryError("You did not specify a(n auto) save path!")

        # If necessary, wait for the previous save to finish.
        save_wait_start_time = time.perf_counter()
        if self.main_save_thread:
            self.main_save_thread.join()
        self.save_wait_time = time.perf_counter() - save_wait_start_time

        # create_main_save_threads makes all the necessary copies in the main thread, while the variables are not being
        # updated. Since the program often freezes when trying to save the model (or a copy thereof) in another thread
        # the model will also be saved in the main thread (in create_main_save_thread).
        self.main_save_thread = self.create_main_save_thread(save_path_without_extension,
                                                             save_model,
                                                             save_training_batch_generator_thread,
                                                             save_validation_batch_generator_thread,
                                                             save_rest_of_learner_thread)
        self.main_save_thread.start()

    def create_main_save_thread(self, save_path_without_extension: str,
                                save_model: bool = True,
                                save_training_batch_generator_thread: threading.Thread = None,
                                save_validation_batch_generator_thread: threading.Thread = None,
                                save_rest_of_learner_thread: threading.Thread = None) -> threading.Thread:
        # At this time save_path_without_extension should be properly set.
        # Note: we cannot simply use (compress_)pickle.dump(self, file, compression="lzma"), because of the threading
        # reasons.

        # Save the model separately as it's big and we don't necessary want to reload it always.
        # As the program will often freeze when saving this in a separate thread, we will save it here, in the main
        # thread.
        if save_model:
            self.save_model_compressed(save_path_without_extension)

        if save_rest_of_learner_thread is None:
            # Create the thread to save the rest.
            # This first makes a deep copy of the variables in this (global) main thread.
            save_rest_of_learner_thread = self.create_save_rest_of_learner_thread(save_path_without_extension)
        return threading.Thread(target=BatchLearner.main_save_thread_method,
                                args=(save_path_without_extension,
                                      save_training_batch_generator_thread,
                                      save_validation_batch_generator_thread,
                                      save_rest_of_learner_thread,
                                      save_model))

    @staticmethod
    def main_save_thread_method(save_path_without_extension: str,
                                save_training_batch_generator_thread: threading.Thread,
                                save_validation_batch_generator_thread: threading.Thread,
                                save_rest_of_learner_thread: threading.Thread,
                                save_model: bool):

        extensions = []

        if save_training_batch_generator_thread:
            save_training_batch_generator_thread.start()
            extensions.append("tbg")
        if save_validation_batch_generator_thread:
            save_validation_batch_generator_thread.start()
            extensions.append("vbg")
        if save_model:
            extensions.append("mo")
        if save_rest_of_learner_thread:  # (should always be the case)
            save_rest_of_learner_thread.start()
            extensions.append("bl")

        # Wait (in the main saving thread, not the global main thread) for these to finish.
        if save_training_batch_generator_thread:
            save_training_batch_generator_thread.join()
        if save_validation_batch_generator_thread:
            save_validation_batch_generator_thread.join()
        if save_rest_of_learner_thread:
            save_rest_of_learner_thread.join()

        # All files are saved as .new<extension>, so as to preserve the current saved file in case the saving
        # process gets interrupted.
        # Now that this process has finished, delete the old .<extension> and rename .new<extension> to .<extension>.
        for ext in extensions:
            old_path = save_path_without_extension + "." + ext
            new_path = save_path_without_extension + ".new" + ext
            if os.path.exists(old_path):
                os.remove(old_path)
            if os.path.exists(new_path):
                os.rename(new_path, old_path)

    def save_model_compressed(self, save_path_without_extension: str):
        # LZ4 instead of LZMA as LZMA takes a very long time to save and does not produce substantially better results:
        # Compression type:  time to save; file size; time to load
        # No compression:        3 205 ms;    969 MB;     4 914 ms
        # LZ4:                   4 787 ms;    702 MB;   110 397 ms
        # LZMA:                451 817 ms;    618 MB;    61 629 ms
        # So unfortunately LZ4 is very slow to load, but since we should hardly every load models, this is less of an
        # issue.

        with lz4.frame.open(save_path_without_extension + ".newmo", "wb") as file:
            pickle.dump(self.model, file)

    def set_up_save_rest_of_learner(self) -> dict:
        # We want to save everything but the batch generators, the model, and the saving thread.
        return copy.deepcopy(
            {var: self.__dict__[var]
                for var in self.__dict__ if var != "training_batch_generator" and var != "validation_batch_generator"
                                            and var != "model" and var != "main_save_thread"})
        # We need a deep copy as otherwise the referenced variables might change in the mean time.
        # (We don't deep copy self.__dict__ and pop model etc., as this would require more memory.)

    def create_save_rest_of_learner_thread(self, save_path_without_extension: str) -> threading.Thread:
        save_dict = self.set_up_save_rest_of_learner()
        return threading.Thread(target=BatchLearner.save_rest_of_learner_vars_dict,
                                args=(save_path_without_extension, save_dict))

    @staticmethod
    def save_rest_of_learner_vars_dict(save_path_without_extension: str, save_dict: dict):
        with open(save_path_without_extension + ".newbl", "wb") as file:
            compress_pickle.dump(save_dict, file, compression="lzma")
        # Saving this rest is actually quite small and fast, so the compression method doesn't really matter too much.

    @classmethod
    def load_saved_batch_learner(cls, saved_learner_path_without_extension: str,
                                 load_model: bool = True, load_batch_generators: bool = True):
        self = cls.__new__(cls)

        with open(saved_learner_path_without_extension + ".bl", "rb") as file:
            self.__dict__ = compress_pickle.load(file, compression="lzma")

        if load_batch_generators:
            self.training_batch_generator = BatchGenerator.load_saved_batch_generator(
                saved_learner_path_without_extension + ".tbg")
            self.validation_batch_generator = BatchGenerator.load_saved_batch_generator(
                saved_learner_path_without_extension + ".vbg")
        else:
            self.training_batch_generator = None
            self.validation_batch_generator = None

        if load_model:
            with lz4.frame.open(saved_learner_path_without_extension + ".mo", "rb") as file:
                self.model = pickle.load(file)
        else:
            self.model = None

        self.main_save_thread = None

        self.is_reloaded = True

        return self

    @staticmethod
    def create_training_batch_generator_for_batch_learner(folder: str, prefix: str, file_nbs: List[int],
                                                          data_loader_class,
                                                          batch_size: int, nb_epochs: int,
                                                          nb_stacked_data: int = None,
                                                          stack_in_last_component: bool = True,
                                                          nb_load_threads: int = 2,
                                                          step: int = 1, offset: int = 0):

        return BatchGenerator(folder, prefix, file_nbs, data_loader_class, batch_size,
                              nb_stacked_data=nb_stacked_data, stack_in_last_component=stack_in_last_component,
                              allow_incomplete_last_batch=True, reset_when_done=False,
                              wrap_back_after_epoch=True, nb_epochs=nb_epochs,
                              nb_load_threads=nb_load_threads,
                              step=step, offset=offset,
                              print_debug=False)

    @staticmethod
    def create_validation_batch_generator_for_batch_learner(folder: str, prefix: str, file_nbs: List[int],
                                                            data_loader_class,
                                                            batch_size: int,
                                                            nb_stacked_data: int = None,
                                                            stack_in_last_component: bool = True,
                                                            nb_load_threads: int = 2,
                                                            step: int = 1, offset: int = 0):

        return BatchGenerator(folder, prefix, file_nbs, data_loader_class, batch_size,
                              allow_incomplete_last_batch=False, reset_when_done=True,
                              wrap_back_after_epoch=True, nb_epochs=1,  # doesn't really matter
                              nb_stacked_data=nb_stacked_data, stack_in_last_component=stack_in_last_component,
                              nb_load_threads=nb_load_threads,
                              step=step, offset=offset,
                              print_debug=False)

    @staticmethod
    def create_random_training_validation_test_split(first_file_nb: int, last_file_nb: int,
                                                     train: int, val: int, test: int,
                                                     step: int = 1)\
            -> (List[int], List[int], List[int]):
        """
        Randomly reorders and splits the inclusive range first_file_nb : step : last_file_nb into three sublists.
        This assumes all numbers between first_file_nb and last_file_nb are valid file numbers.

        The arguments train, val and test refer to the weight (unnormalized size) of the divide. The resulting
        relative frequencies are train / (train + val + test) etc.
        """

        total_weight = train + val + test
        scrambled = list(range(first_file_nb, last_file_nb + 1, step))
        random.shuffle(scrambled)

        train_file_nbs = scrambled[0 : round(len(scrambled) * train / total_weight)]
        val_file_nbs = scrambled[len(train_file_nbs) : len(train_file_nbs) + round(len(scrambled) * val / total_weight)]
        test_file_nbs = scrambled[len(train_file_nbs) + len(val_file_nbs) : ]

        return train_file_nbs, val_file_nbs, test_file_nbs
