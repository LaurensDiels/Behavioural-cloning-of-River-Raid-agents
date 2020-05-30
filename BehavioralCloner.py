# For gpu support using pip:
# *Install package tensorflow-gpu
# *Install cuda toolkit 10.0 from Nvidia (or 10.2 and manually install cudart64_100.dll and rename cublas64_10.dll to cublas64_100.dll)
# *Install cuDNN from Nvidia
# (i.e. just use conda)
import math
import sys
import random
import pickle
from typing import Callable, List

import compress_pickle
import numpy

import keras
from keras import regularizers
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras import utils

import hyperopt
from hyperopt import fmin, tpe, hp
from hyperopt import Trials

from GlobalSettings import *
from DataLoader import EpisodeDataLoader
from BatchGenerator import NB_CLASSES
from BatchLearner import BatchLearner
from BatchLearnerSmoothingOptions import BatchLearnerSmoothingOptions


SEED = 31415
numpy.random.seed(SEED)  # For reproducibility
random.seed(SEED)

###################
# Settings
CONFIRM_SETTINGS = True  # Turn off at the VSC (and for tuning)
####################


def normalize_data(x: numpy.ndarray, y: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
    normalized_batch_x = x.astype(NORMALISED_SCREEN_DATA_TYPE) / MAX_PIXEL_VALUE
    normalized_batch_y = utils.to_categorical(y, NB_CLASSES)
    return normalized_batch_x, normalized_batch_y


def add_L2_regularization(model: keras.Model, regularization_factor: float):
    """To be called before the model is compiled."""
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = regularizers.l2(regularization_factor)


def create_VGG_model_transfer_learning_MCDropout(input_shape, learning_rate, show_summary=True):
    vgg_no_top = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in vgg_no_top.layers:
        layer.trainable = False

    x = Flatten()(vgg_no_top.output)
    x = Dropout(0.5)(x, training=True)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x, training=True)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x, training=True)
    x = Dense(NB_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs=vgg_no_top.inputs, outputs=x)

    if show_summary:
        model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    return model


def create_VGG_model_transfer_learning_MCDropout_quarter_dense(input_shape, learning_rate, show_summary=True):
    vgg_no_top = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in vgg_no_top.layers:
        layer.trainable = False

    x = Flatten()(vgg_no_top.output)
    x = Dropout(0.5)(x, training=True)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x, training=True)
    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x, training=True)
    x = Dense(NB_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs=vgg_no_top.inputs, outputs=x)

    if show_summary:
        model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    return model


def create_custom_model_a(input_shape, learning_rate,
                          regularization_constant=None, batch_norm=False,
                          show_summary=True):
    inputs = Input(shape=input_shape)
    if batch_norm:
        x = BatchNormalization()(inputs)
    else:
        x = inputs
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(8, 8))(x)
    x = Dropout(0.5)(x, training=True)
    x = Flatten()(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = Dropout(0.5)(x, training=True)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dense(128)(x)
    x = Dropout(0.5)(x, training=True)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dense(NB_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)

    if regularization_constant:
        add_L2_regularization(model, regularization_constant)

    if show_summary:
        model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(learning_rate),
                  metrics=['accuracy'])

    return model


def create_custom_model_b(input_shape, learning_rate,
                          regularization_constant=None, batch_norm=False,
                          show_summary=True):
    inputs = Input(shape=input_shape)
    if batch_norm:
        x = BatchNormalization()(inputs)
    else:
        x = inputs
    x = Conv2D(128, kernel_size=(5, 5), activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.5)(x, training=True)
    x = Flatten()(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dense(64)(x)
    x = Dropout(0.5)(x, training=True)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dense(64)(x)
    x = Dropout(0.5)(x, training=True)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Dense(NB_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)

    if regularization_constant:
        add_L2_regularization(model, regularization_constant)

    if show_summary:
        model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(learning_rate),
                  metrics=['accuracy'])

    return model


def create_custom_model_c(input_shape, learning_rate, regularization_constant, show_summary=True):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=(5, 5), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x, training=True)
    x = Conv2D(128, kernel_size=(5, 5), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x, training=True)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.5)(x, training=True)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Dropout(0.5)(x, training=True)
    x = Dense(64)(x)
    x = Dropout(0.5)(x, training=True)
    x = Dense(NB_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)

    add_L2_regularization(model, regularization_constant)

    if show_summary:
        model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(learning_rate),
                  metrics=['accuracy'])

    return model


def create_custom_model_d(input_shape, learning_rate, show_summary=True):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(5, 5), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(32, 32))(x)
    x = Dropout(0.5)(x, training=True)
    x = Flatten()(x)
    x = Dense(32)(x)
    x = Dropout(0.5)(x, training=True)
    x = Dense(32)(x)
    x = Dropout(0.5)(x, training=True)
    x = Dense(NB_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)

    if show_summary:
        model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(learning_rate),
                  metrics=['accuracy'])

    return model


def train_model(episodes_folder: str, fraction_training_validation: int,
                training_file_nbs: List[int], validation_file_nbs: List[int], test_file_nbs: List[int],
                model_creation_function: Callable,
                batch_size: int = None, nb_epochs: int = None, step: int = None,
                nb_frames_to_stack: int = None, learning_rate: float = None, regularization_constant: float = None,
                name: str = None, save_folder: str = None,
                save_batch_learner: bool = None, save_model: bool = None, save_checkpoints: bool = None,
                save_data_split: bool = None,
                checkpoint_folder_name: str = None, checkpoint_file_name: str = None,
                make_plots: bool = None,
                print_outputs: bool = None, print_debug: bool = None)\
        -> BatchLearner:
    """model_creation_function has to take in three arguments (with non-default value):
    the input shape, the learning rate and the regularization constant. It should also have a boolean input
    show_summary. The output needs to be a compiled Keras model.

    Using save_batch_learner=True we can retrieve among others the training history.

    The checkpoint folder will be a subfolder of save_folder.
    """

    # Default values
    # (By using None these can be propagated to functions calling train_model. Otherwise in e.g. train_model_b we would
    # also have to hardcode the default values.)
    if save_batch_learner is None:
        save_batch_learner = True
    if save_model is None:
        save_model = True
    if save_checkpoints is None:
        save_checkpoints = True
    if save_data_split is None:
        save_data_split = True
    if batch_size is None:
        batch_size = 32
    if nb_epochs is None:
        nb_epochs = 25
    if step is None:
        step = 3
    if nb_frames_to_stack is None:
        nb_frames_to_stack = 3
    if learning_rate is None:
        learning_rate = 1e-4
    if regularization_constant is None:
        regularization_constant = 1e-3
    if name is None:
        name = "BatchLearner"
    if save_folder is None:
        save_folder = BC_TEST_MODEL_FOLDER
    if checkpoint_folder_name is None:
        checkpoint_folder_name = "Checkpoint"
    if checkpoint_file_name is None:
        checkpoint_file_name = "Checkpoint"
    if make_plots is None:
        make_plots = True
    if print_outputs is None:
        print_outputs = True
    if print_debug is None:
        print_debug = True

    batch_learner = BatchLearner(fraction_training_validation, normalize_data,
                                 BatchLearner.create_training_batch_generator_for_batch_learner(
                                     episodes_folder, EPISODE_PREFIX, training_file_nbs, EpisodeDataLoader,
                                     batch_size, nb_epochs,
                                     nb_stacked_data=nb_frames_to_stack,
                                     stack_in_last_component=True,
                                     nb_load_threads=2,
                                     step=step, offset=0
                                 ),
                                 BatchLearner.create_validation_batch_generator_for_batch_learner(
                                     episodes_folder, EPISODE_PREFIX, validation_file_nbs, EpisodeDataLoader,
                                     batch_size,
                                     nb_stacked_data=nb_frames_to_stack,
                                     stack_in_last_component=True,
                                     nb_load_threads=2,
                                     step=step, offset=0
                                 ),
                                 auto_save_folder=os.path.join(save_folder, checkpoint_folder_name) if save_checkpoints
                                                                                                    else None,
                                 auto_save_file_name=checkpoint_file_name if save_checkpoints else None,
                                 save_after_every_n_epochs=1 if save_checkpoints else None,
                                 smoothing_option=BatchLearnerSmoothingOptions.SIMPLE_MOVING_AVERAGE_EPOCH_WINDOW,
                                 print_outputs=print_outputs, print_debug=print_debug)

    model = model_creation_function(batch_learner.get_input_shape(), learning_rate, regularization_constant,
                                    show_summary=False)
    # Don't show the summary here, but in confirm_settings (if enabled).

    if CONFIRM_SETTINGS:
        confirm_settings(episodes_folder=episodes_folder,
                         fraction_training_validation=fraction_training_validation,
                         training_file_nbs=training_file_nbs,
                         validation_file_nbs=validation_file_nbs,
                         test_file_nbs=test_file_nbs,
                         model_creation_function=model_creation_function,
                         model=model,
                         batch_size=batch_size,
                         nb_epochs=nb_epochs, step=step,
                         nb_frames_to_stack=nb_frames_to_stack,
                         learning_rate=learning_rate,
                         regularization_constant=regularization_constant,
                         name=name,
                         save_folder=save_folder,
                         save_batch_learner=save_batch_learner,
                         save_model=save_model,
                         save_checkpoints=save_checkpoints,
                         save_data_split=save_data_split,
                         checkpoint_folder=checkpoint_folder_name,
                         checkpoint_file_name=checkpoint_file_name,
                         make_plots=make_plots,
                         print_outputs=print_outputs,
                         print_debug=print_debug)

    if save_data_split:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        split = {"train": training_file_nbs, "val": validation_file_nbs, "test": test_file_nbs}
        with open(os.path.join(save_folder, name + "_SplitFileNbs.pcl"), "wb") as file:
            pickle.dump(split, file)

    batch_learner.start_training_model(model)
    finish_training(batch_learner, name, save_folder, save_batch_learner, save_model, make_plots)

    return batch_learner


def finish_training(batch_learner: BatchLearner,  name: str, save_folder: str,
                    save_batch_learner: bool, save_model: bool, make_plots: bool):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if save_model:
        batch_learner.model.save(os.path.join(save_folder, name + "_model.h5"))

    if save_batch_learner:
        batch_learner.save(save_path_without_extension=os.path.join(save_folder, name),
                           save_model=False, save_batch_generators=False)

    if make_plots:
        batch_learner.make_history_plot(os.path.join(save_folder, name + "_TrainHistory.svg"),
                                        title="", ylim={"accuracy": (0, 1), "loss": (0, 5)})

        batch_learner.make_history_plot(os.path.join(save_folder, name + "_TrainHistoryNoSmoothing.svg"),
                                        title="", use_smoothing=False,
                                        ylim={"accuracy": (0, 1), "loss": (0, 5)})

    # We could also remove the checkpoint now. But just to be safe we keep it and delete it manually later.


def resume_training(batch_learner_name, save_folder, checkpoint_folder_name, checkpoint_file_name,
                    save_batch_learner: bool = True, save_model: bool = True, make_plots=True):
    """In case the training interrupted for any reason (e.g. manually, or because of a crash)"""
    batch_learner = BatchLearner.load_saved_batch_learner(
        saved_learner_path_without_extension=os.path.join(save_folder, checkpoint_folder_name, checkpoint_file_name),
        load_model=True, load_batch_generators=True)
    batch_learner.continue_training_model()

    finish_training(batch_learner, batch_learner_name, save_folder, save_batch_learner, save_model, make_plots)


def confirm_settings(**kwargs):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Are you sure you want to use the following settings:")
    print("")
    for key, value in kwargs.items():
        if not (key.endswith("file_nbs") or key == "model"):
            print("{} = {}".format(key, value))
        elif key.endswith("file_nbs"):
            if value:
                nbs_len = len(value)
                nbs_start = value[0]
                nbs_last = value[-1]
            else:  # None or empty
                nbs_len = 0
                nbs_start = "/"
                nbs_last = "/"
            print("{}: len = {}, first = {}, last = {}".format(key,
                                                               nbs_len,
                                                               nbs_start,
                                                               nbs_last))
    if "model" in kwargs:  # print as last
        print("model summary: ")
        print("")
        kwargs["model"].summary()

    print("")
    print("Use these settings? (y/n)")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    x = input()
    if x != "y":
        print("Did not receive \"y\". Aborting.")
        sys.exit()


def tune_model_a(args):
    """args needs to contain the learning rate, the regularization strength and the number of frames to stack
     (i.e. the hyperparameters to tune), as well as the folder containing the (training/validation/test) episodes,
     the folder to save the results, and the training/validation file numbers of which a random subset of
     TUNING_NB_TRAINING_EPISODES/TUNING_NB_VALIDATION_EPISODES will be used.
     (The parameters need to be in that order.)"""
    learning_rate, regularization_strength, nb_frames_to_stack, \
        episodes_folder, save_folder, \
        potential_training_file_nbs, potential_validation_file_nbs = args

    name = "LR={} - RC={} - SF={}".format(learning_rate, regularization_strength, nb_frames_to_stack)
    print("////////////////////////////////////////////////")
    print(name)
    print("////////////////////////////////////////////////")

    NB_EPOCHS = 10
    STEP = 3

    training_file_nbs = list(potential_training_file_nbs)
    validation_file_nbs = list(potential_validation_file_nbs)
    random.shuffle(training_file_nbs)
    random.shuffle(validation_file_nbs)
    training_file_nbs = training_file_nbs[:TUNING_NB_TRAINING_EPISODES]
    validation_file_nbs = validation_file_nbs[:TUNING_NB_VALIDATION_EPISODES]

    batch_learner = train_model(episodes_folder, TUNING_FRACTION_TRAINING_VALIDATION,
                                training_file_nbs, validation_file_nbs, [],
                                create_custom_model_a,
                                nb_epochs=NB_EPOCHS, step=STEP,
                                nb_frames_to_stack=nb_frames_to_stack, learning_rate=learning_rate,
                                regularization_constant=regularization_strength,
                                name=name,
                                save_folder=save_folder,
                                save_batch_learner=True, save_model=False, save_checkpoints=False, save_data_split=True,
                                make_plots=False)

    return {
            'loss': -batch_learner.validation_history['accuracy'][-1],  # try to minimize the loss
            'status': hyperopt.STATUS_OK,
            'eval_time': batch_learner.get_total_runtime(),
            }


def tune_custom_model_a_hyperparameters(episodes_folder: str, save_folder: str,
                                        potential_training_file_nbs: List[int],
                                        potential_validation_file_nbs: List[int],
                                        cpickled_trials_path: str = None):
    """cpickled_trials_path can be used to resume the tuning. By default it will be in the save_folder and have the
    file name Trials.xz (as we use lzma-compression with compress-pickle)."""
    if cpickled_trials_path is None:
        cpickled_trials_path = os.path.join(save_folder, "Trials.xz")

    if not os.path.exists(cpickled_trials_path):
        trials = Trials()
        current_nb_runs = 0
    else:
        with open(cpickled_trials_path, 'rb') as file:
            trials = compress_pickle.load(file, compression="lzma")
        current_nb_runs = len(trials.trials)

    best_hyperparameters = None
    while current_nb_runs < TUNING_NB_RUNS:
        best_hyperparameters = fmin(tune_model_a,
                                    space=(hp.loguniform('learning_rate', math.log(10**-5), math.log(10**-3)),
                                           hp.loguniform('regularization_strength', math.log(10**-4), math.log(10**-2)),
                                           hp.uniformint('nb_frames_to_stack', 2, 25),
                                           hp.choice('episodes_folder', [episodes_folder]),  # not really a choice
                                           hp.choice('save_folder', [save_folder]), # just a way to pass more parameters
                                           hp.choice('potential_training_file_nbs', [potential_training_file_nbs]),
                                           hp.choice('potential_validation_file_nbs', [potential_validation_file_nbs])),
                                    algo=tpe.suggest,
                                    max_evals=current_nb_runs+1,  # just keep going (Note: messes with the progress bar)
                                    trials=trials)
        current_nb_runs += 1  # (after the += 1: == len(trials.trials))

        # Save after every tuning run
        with open(cpickled_trials_path, "wb") as file:
            compress_pickle.dump(trials, file, compression="lzma")

    print(best_hyperparameters)
    print(trials.best_trial["result"]["loss"])


def tune_custom_model_a_hyperparameters_RL():
    total_nb_files = len(os.listdir(FIXED_RL_EPISODES_FOLDER))
    potential_training_file_nbs = list(range(RL_BC_TRAINING_START_FILE_NB, RL_BC_VALIDATION_START_FILE_NB))
    potential_validation_file_nbs = list(range(RL_BC_VALIDATION_START_FILE_NB, total_nb_files))
    tune_custom_model_a_hyperparameters(FIXED_RL_EPISODES_FOLDER, RL_BC_HYPERPARAMETER_TUNING_FOLDER,
                                        potential_training_file_nbs, potential_validation_file_nbs)


def tune_custom_model_a_hyperparameters_human():
    potential_training_file_nbs, potential_validation_file_nbs, test_file_nbs = get_human_episodes_split()
    tune_custom_model_a_hyperparameters(HUMAN_EPISODES_FOLDER, HUMAN_BC_HYPERPARAMETER_TUNING_FOLDER,
                                        potential_training_file_nbs, potential_validation_file_nbs)


def tune_custom_model_a_hyperparameters_random():
    total_nb_files = len(os.listdir(RANDOM_EPISODES_FOLDER))
    potential_training_file_nbs = list(range(RANDOM_BC_TRAINING_START_FILE_NB, RANDOM_BC_VALIDATION_START_FILE_NB))
    potential_validation_file_nbs = list(range(RANDOM_BC_VALIDATION_START_FILE_NB, total_nb_files))
    tune_custom_model_a_hyperparameters(RANDOM_EPISODES_FOLDER, RANDOM_BC_HYPERPARAMETER_TUNING_FOLDER,
                                        potential_training_file_nbs, potential_validation_file_nbs)


def train_tuned_custom_model_a_RL_varying_nb_of_episodes(nb_training_episodes: int):
    NB_EPOCHS = 25
    STEP = 3

    training_file_nbs = list(range(nb_training_episodes))
    validation_file_nbs = list(range(RL_BC_VALIDATION_START_FILE_NB,
                                     math.ceil(RL_BC_VALIDATION_START_FILE_NB
                                               + nb_training_episodes / RL_BC_FRACTION_TRAINING_VALIDATION)))

    name = "Training{}Episodes".format(nb_training_episodes)
    save_folder = os.path.join(FIXED_RL_BC_VARYING_NB_EPISODES_BASE_FOLDER, str(nb_training_episodes))

    train_model(FIXED_RL_EPISODES_FOLDER, RL_BC_FRACTION_TRAINING_VALIDATION,
                training_file_nbs, validation_file_nbs, [],
                create_custom_model_a,
                nb_epochs=NB_EPOCHS, step=STEP,
                nb_frames_to_stack=CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_NB_FRAMES_TO_STACK,
                learning_rate=CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_LEARNING_RATE,
                regularization_constant=CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_REGULARIZATION_CONSTANT,
                name=name, save_folder=save_folder,
                save_batch_learner=True, save_model=True, save_checkpoints=True, save_data_split=True,
                make_plots=True
                )


def train_tuned_custom_model_a_random():
    NB_EPOCHS = 25
    STEP = 3

    training_file_nbs = list(range(RANDOM_BC_TRAINING_START_FILE_NB, RANDOM_BC_VALIDATION_START_FILE_NB))
    validation_file_nbs = list(range(RANDOM_BC_VALIDATION_START_FILE_NB, RANDOM_BC_VALIDATION_END_FILE_NB))

    train_model(RANDOM_EPISODES_FOLDER, RANDOM_BC_FRACTION_TRAINING_VALIDATION,
                training_file_nbs, validation_file_nbs, [],
                create_custom_model_a,
                nb_epochs=NB_EPOCHS, step=STEP,
                nb_frames_to_stack=CUSTOM_A_TUNED_HYPERPARAMETERS_RANDOM_BC_NB_FRAMES_TO_STACK,
                learning_rate=CUSTOM_A_TUNED_HYPERPARAMETERS_RANDOM_BC_LEARNING_RATE,
                regularization_constant=CUSTOM_A_TUNED_HYPERPARAMETERS_RANDOM_BC_REGULARIZATION_CONSTANT,
                name="RandomBC", save_folder=RANDOM_BC_MODEL_FOLDER,
                save_batch_learner=True, save_model=True, save_checkpoints=True, save_data_split=True,
                make_plots=True
                )


def get_human_episodes_split() -> (List[int], List[int], List[int]):
    """Allows for consistency in the splits for multiple models."""
    if os.path.exists(HUMAN_BC_CUSTOM_A_MODEL_FILE_SPLIT_PATH):
        with open(HUMAN_BC_CUSTOM_A_MODEL_FILE_SPLIT_PATH, "rb") as file:
            split = pickle.load(file)
        training_file_nbs = split["train"]
        validation_file_nbs = split["val"]
        test_file_nbs = split["test"]
    else:
        training_file_nbs, validation_file_nbs, test_file_nbs = \
            BatchLearner.create_random_training_validation_test_split(0, len(os.listdir(HUMAN_EPISODES_FOLDER)) - 1,
                                                                      HUMAN_BC_FRACTION_TRAINING_VALIDATION, 1, 1)
        split = {"train": training_file_nbs, "val": validation_file_nbs, "test": test_file_nbs}
        split_file_folder = os.path.dirname(HUMAN_BC_CUSTOM_A_MODEL_FILE_SPLIT_PATH)
        if not os.path.exists(split_file_folder):
            os.makedirs(split_file_folder)
        with open(HUMAN_BC_CUSTOM_A_MODEL_FILE_SPLIT_PATH, "wb") as file:
            pickle.dump(split, file)

    return training_file_nbs, validation_file_nbs, test_file_nbs


def train_tuned_custom_model_a_human(create_custom_model_a_method: Callable, name: str = None, save_folder: str = None,
                                     nb_frames_to_stack: int = None):
    if nb_frames_to_stack is None:
        nb_frames_to_stack = CUSTOM_A_TUNED_HYPERPARAMETERS_HUMAN_BC_NB_FRAMES_TO_STACK

    NB_EPOCHS = 25
    STEP = 3

    training_file_nbs, validation_file_nbs, test_file_nbs = get_human_episodes_split()

    train_model(HUMAN_EPISODES_FOLDER, HUMAN_BC_FRACTION_TRAINING_VALIDATION,
                training_file_nbs, validation_file_nbs, test_file_nbs,
                create_custom_model_a_method,
                save_batch_learner=True, save_model=True, save_checkpoints=True, save_data_split=True,
                nb_epochs=NB_EPOCHS, step=STEP,
                nb_frames_to_stack=nb_frames_to_stack,
                learning_rate=CUSTOM_A_TUNED_HYPERPARAMETERS_HUMAN_BC_LEARNING_RATE,
                regularization_constant=CUSTOM_A_TUNED_HYPERPARAMETERS_HUMAN_BC_REGULARIZATION_CONSTANT,
                name=name, save_folder=save_folder)


def train_tuned_custom_model_a_human_scratch():
    train_tuned_custom_model_a_human(create_custom_model_a, "Human_Scratch", HUMAN_BC_CUSTOM_A_MODEL_SCRATCH_FOLDER)


def train_tuned_custom_model_a_human_transfer_learning_init():
    train_tuned_custom_model_a_human(create_model_method_load_RLBC_model, "Human_TL_Init",
                                     HUMAN_BC_CUSTOM_A_MODEL_TL_INIT_FOLDER,
                                     CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_NB_FRAMES_TO_STACK)
                                     # Note: RL_BC to fit the shapes


def create_model_method_load_RLBC_model(input_shape, learning_rate, regularization_constant, show_summary):
    model = keras.models.load_model(RL_BC_VARYING_NB_EPISODES_MODEL_PATHS[-1])  # maximum number of episodes
    return model


def train_tuned_custom_model_a_human_transfer_learning_reset_top_rest_frozen():
    train_tuned_custom_model_a_human(create_model_method_load_RLBC_model_reset_top_rest_frozen, "Human_TL_Freeze",
                                     HUMAN_BC_CUSTOM_A_MODEL_TL_FREEZE_FOLDER,
                                     CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_NB_FRAMES_TO_STACK)


def create_model_method_load_RLBC_model_reset_top_rest_frozen(input_shape, learning_rate, regularization_constant,
                                                              show_summary):
    base_model = keras.models.load_model(RL_BC_VARYING_NB_EPISODES_MODEL_PATHS[-1])  # maximum number of episodes

    # We will retrain only the last three layers
    x = Dense(128, kernel_regularizer=regularizers.l2(regularization_constant),
              name="new_dense_2")(base_model.layers[-4].output)
    # If we don't manually add the names, the automatically generated names (e.g. dense_1) will clash with the ones
    # from base_model.
    x = Dropout(0.5, name="new_dropout_3")(x, training=True)
    x = Dense(NB_CLASSES, kernel_regularizer=regularizers.l2(regularization_constant),
              activation='softmax', name="new_dense_3")(x)
    model = Model(inputs=base_model.input, outputs=x)
    # So freeze all others
    for layer in model.layers[:-3]:
        layer.trainable = False

    if show_summary:
        model.summary()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    return model


def train_tuned_custom_model_a_human_transfer_learning_reset_top_finetune():
    train_tuned_custom_model_a_human(create_model_method_load_RLBC_model_reset_top_finetune, "Human_TL_Finetune",
                                     HUMAN_BC_CUSTOM_A_MODEL_TL_FINETUNE_FOLDER,
                                     CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_NB_FRAMES_TO_STACK)


def create_model_method_load_RLBC_model_reset_top_finetune(input_shape, learning_rate, regularization_constant,
                                                           show_summary):
    model = keras.models.load_model(HUMAN_BC_CUSTOM_A_MODEL_TL_FREEZE_MODEL_PATH)
    # Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    
    if show_summary:
        model.summary()

    return model


if __name__ == '__main__':
    pass

    # tune_custom_model_a_hyperparameters_RL()  # -> now set CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC__... in GlobalSettings.py
    # for nb_episodes in RL_BC_VARYING_NB_EPISODES_NBS:
    #     train_tuned_custom_model_a_RL_varying_nb_of_episodes(nb_episodes)

    #train_fixed_tuned_custom_model_b_RL_2000_episodes()

    #tune_custom_model_a_hyperparameters_random()  # -> now set CUSTOM_A_TUNED_HYPERPARAMETERS_RANDOM__... in GlobalSettings.py
    #train_tuned_custom_model_a_random()

    #tune_custom_model_a_hyperparameters_human()  # -> now set CUSTOM_A_TUNED_HYPERPARAMETERS_HUMAN__... in GlobalSettings.py
    #train_tuned_custom_model_a_human_scratch()
    #train_tuned_custom_model_a_human_transfer_learning_init()
    #train_tuned_custom_model_a_human_transfer_learning_reset_top_rest_frozen()
    #train_tuned_custom_model_a_human_transfer_learning_reset_top_finetune()


