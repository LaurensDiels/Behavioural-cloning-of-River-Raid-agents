import os

# More constants than settings
ATARI_FPS = 60.0
MAX_PIXEL_VALUE = 255
NORMALISED_SCREEN_DATA_TYPE = "float32"

####################
# Global settings:

EPISODE_PREFIX = "Ep"
HUMAN_EPISODES_FOLDER = os.path.join("Data", "HumanBC", "Episodes")
FIXED_RL_EPISODES_FOLDER = os.path.join("Data", "RLBC", "FixedEpisodes")
RANDOM_EPISODES_FOLDER = os.path.join("Data", "RandomBC", "Episodes")

BC_TEST_MODEL_FOLDER = os.path.join("Data", "TestBC")

TUNING_FRACTION_TRAINING_VALIDATION = 2
TUNING_NB_TRAINING_EPISODES = 50
TUNING_NB_VALIDATION_EPISODES = 25
TUNING_NB_RUNS = 50

HUMAN_BC_FRACTION_TRAINING_VALIDATION = 8  # 80% training, 10% validation, 10% test
# (high because of limited data, test needs to be enforced as we cannot just create more data)
RL_BC_FRACTION_TRAINING_VALIDATION = 4  # 80% training, 20% validation, no explicit test
RL_BC_TRAINING_START_FILE_NB = 0
RL_BC_VALIDATION_START_FILE_NB = 2000
# There are 2572 episodes in total. Since we use 20% validation, we can go up to (about) 2000 training episodes.
RANDOM_BC_FRACTION_TRAINING_VALIDATION = 4  # cf. RL_BC
RANDOM_BC_TRAINING_START_FILE_NB = 0
RANDOM_BC_VALIDATION_START_FILE_NB = 400
RANDOM_BC_VALIDATION_END_FILE_NB = 500
# There are 510 episodes.


RL_BC_HYPERPARAMETER_TUNING_FOLDER = os.path.join("Data", "RLBC", "BCModel", "Custom A Tuning")
HUMAN_BC_HYPERPARAMETER_TUNING_FOLDER = os.path.join("Data", "HumanBC", "BCModel", "Custom A Tuning")
RANDOM_BC_HYPERPARAMETER_TUNING_FOLDER = os.path.join("Data", "RandomBC", "BCModel", "Tuning")

# The values below (mistakenly) used overall average accuracy as performance metric.
OLD_CUSTOM_B_TUNED_HYPERPARAMETERS_RL_BC_LEARNING_RATE = 0.00040344172495713777
OLD_CUSTOM_B_TUNED_HYPERPARAMETERS_RL_BC_REGULARIZATION_CONSTANT = 0.0002426903941085837
OLD_CUSTOM_B_TUNED_HYPERPARAMETERS_RL_BC_NB_FRAMES_TO_STACK = 3

# These values correctly use the validation accuracy averaged over the last epoch as metric. But the search process was
# the same as for the mistaken values above, so that the Bayesian search might have been lead in the wrong direction.
OLD_CUSTOM_B_TUNED_HYPERPARAMETERS_RL_BC_LEARNING_RATE_FIXED = 0.00028072551713997733
OLD_CUSTOM_B_TUNED_HYPERPARAMETERS_RL_BC_REGULARIZATION_CONSTANT_FIXED = 0.00017467252862900996
OLD_CUSTOM_B_TUNED_HYPERPARAMETERS_RL_BC_NB_FRAMES_TO_STACK_FIXED = 4


CUSTOM_A_TUNED_HYPERPARAMETERS_HUMAN_BC_LEARNING_RATE = 0.0002083644915002709
CUSTOM_A_TUNED_HYPERPARAMETERS_HUMAN_BC_REGULARIZATION_CONSTANT = 0.0003107004765384462
CUSTOM_A_TUNED_HYPERPARAMETERS_HUMAN_BC_NB_FRAMES_TO_STACK = 4

CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_LEARNING_RATE = 0.00021138987507268324
CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_REGULARIZATION_CONSTANT = 0.007119949294804445
CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_NB_FRAMES_TO_STACK = 2


CUSTOM_A_TUNED_HYPERPARAMETERS_RANDOM_BC_LEARNING_RATE = 1.84175924810216e-05
CUSTOM_A_TUNED_HYPERPARAMETERS_RANDOM_BC_REGULARIZATION_CONSTANT = 0.00146181171746405
CUSTOM_A_TUNED_HYPERPARAMETERS_RANDOM_BC_NB_FRAMES_TO_STACK = 10


HUMAN_BC_VGGMODEL_PATH = os.path.join("Data", "HumanBC", "BCModel", "VGG", "model.h5")
HUMAN_BC_VGGREGMODEL_PATH = os.path.join("Data", "HumanBC", "BCModel", "VGG reg", "model.h5")

HUMAN_BC_CUSTOM_A_MODEL_SCRATCH_FOLDER = os.path.join("Data", "HumanBC", "BCModel", "Custom A Scratch")
HUMAN_BC_CUSTOM_A_MODEL_SCRATCH_MODEL_PATH = os.path.join(HUMAN_BC_CUSTOM_A_MODEL_SCRATCH_FOLDER,
                                                          "Human_Scratch_model.h5")
HUMAN_BC_CUSTOM_A_MODEL_TL_INIT_FOLDER = os.path.join("Data", "HumanBC", "BCModel", "Custom A TL Init")
HUMAN_BC_CUSTOM_A_MODEL_TL_INIT_MODEL_PATH = os.path.join(HUMAN_BC_CUSTOM_A_MODEL_TL_INIT_FOLDER,
                                                          "Human_TL_Init_model.h5")

HUMAN_BC_CUSTOM_A_MODEL_TL_FREEZE_FOLDER = os.path.join("Data", "HumanBC", "BCModel", "Custom A TL Freeze")
HUMAN_BC_CUSTOM_A_MODEL_TL_FREEZE_MODEL_PATH = os.path.join(HUMAN_BC_CUSTOM_A_MODEL_TL_FREEZE_FOLDER,
                                                            "Human_TL_Freeze_model.h5")

HUMAN_BC_CUSTOM_A_MODEL_TL_FINETUNE_FOLDER = os.path.join("Data", "HumanBC", "BCModel", "Custom A TL Finetune")
HUMAN_BC_CUSTOM_A_MODEL_TL_FINETUNE_MODEL_PATH = os.path.join(HUMAN_BC_CUSTOM_A_MODEL_TL_FINETUNE_FOLDER,
                                                              "Human_TL_Finetune_model.h5")

HUMAN_BC_CUSTOM_A_MODEL_FILE_SPLIT_PATH = os.path.join("Data", "HumanBC", "BCModel", "SplitFileNbs.pcl")


RL_BC_VGGMODEL_PATH = os.path.join("Data", "RLBC", "BCModel", "VGG", "model.h5")
RL_BC_VGGQDMODEL_PATH = os.path.join("Data", "RLBC", "BCModel", "VGG quarter dense", "model.h5")
RL_BC_HVGGMODEL_PATH = os.path.join("Data", "RLBC", "BCModel", "Custom A", "model.h5")

RL_BC_VARYING_NB_EPISODES_BASE_FOLDER = os.path.join("Data", "RLBC", "BCModel", "VaryingNbEpisodes")
RL_BC_VARYING_NB_EPISODES_NBS = [1, 2, 5, 10, 25, 50, 125, 250, 500, 1000, 2000]
RL_BC_VARYING_NB_EPISODES_BASE_MODEL_FILE_NAME = "Training{}Episodes_model.h5"
RL_BC_VARYING_NB_EPISODES_MODEL_PATHS = [os.path.join(RL_BC_VARYING_NB_EPISODES_BASE_FOLDER, str(nb),
                                                      RL_BC_VARYING_NB_EPISODES_BASE_MODEL_FILE_NAME.format(nb))
                                         for nb in RL_BC_VARYING_NB_EPISODES_NBS]


RANDOM_BC_MODEL_FOLDER = os.path.join("Data", "RandomBC", "BCModel", "Model")
RANDOM_BC_MODEL_PATH = os.path.join(RANDOM_BC_MODEL_FOLDER, "RandomBC_model.h5")


RL_MODEL_FOLDER_NAME = "Models"
RL_GIFS_FOLDER_NAME = "TrainingGifs"
RL_PI_PREDICT_MODEL_WEIGHTS_BASE_FILE_NAME = "pi_predictModelWeights"
RL_V_PREDICT_MODEL_WEIGHTS_BASE_FILE_NAME = "v_predictModelWeights"


FIXED_RL_NB_TRAINING_EPISODES = 1000
FIXED_RL_BASE_FOLDER = os.path.join("Data", "RL", "Fixed")
FIXED_RL_PI_PREDICT_MODEL_WEIGHTS_PATH = os.path.join(FIXED_RL_BASE_FOLDER, RL_MODEL_FOLDER_NAME,
                                                      RL_PI_PREDICT_MODEL_WEIGHTS_BASE_FILE_NAME
                                                      + str(FIXED_RL_NB_TRAINING_EPISODES) + ".h5")
FIXED_RL_V_PREDICT_MODEL_WEIGHTS_PATH = os.path.join(FIXED_RL_BASE_FOLDER, RL_MODEL_FOLDER_NAME,
                                                     RL_V_PREDICT_MODEL_WEIGHTS_BASE_FILE_NAME
                                                     + str(FIXED_RL_NB_TRAINING_EPISODES) + ".h5")

IMPROVING_SIMPLE_RL_EPISODES_FOLDER = os.path.join("Data", "RLBC", "ImprovingEpisodes", "Simple")
IMPROVING_SIMPLE_RL_BASE_FOLDER = os.path.join("Data", "RL", "Improving", "Simple")

IMPROVING_GREEDY_RL_EPISODES_FOLDER = os.path.join("Data", "RLBC", "ImprovingEpisodes", "Greedy")
IMPROVING_GREEDY_RL_BASE_FOLDER = os.path.join("Data", "RL", "Improving", "Greedy")

FIXED_RL_BC_VARYING_NB_EPISODES_BASE_FOLDER = os.path.join("Data", "RLBC", "BCModel", "VaryingNbEpisodes")


RL_PREPROCESS_HEIGHT = 105
RL_PREPROCESS_WIDTH = 80
RL_PREPROCESS_GRAYSCALE = True
RL_PREPROCESS_NUM_FRAMES = 3
RL_PI_UPDATE_STRATEGY = "ppo"


MODEL_DISTRIBUTION_COMPARISON_FOLDER = os.path.join("ModelComparisons", "Distribution")
MODEL_SCORES_TIME_ALIVE_COMPARISON_FOLDER = os.path.join("ModelComparisons", "Scores and time alive")
AGENTS_VARIABILITY_COMPARISON_FOLDER = os.path.join("ModelComparisons", "Variability")

####################
