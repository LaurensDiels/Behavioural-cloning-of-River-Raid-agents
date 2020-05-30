from enum import Enum, auto


class BatchLearnerSmoothingOptions(Enum):
    NO_SMOOTHING = auto()
    RUNNING_AVERAGE = auto()
    RUNNING_AVERAGE_RESET_AFTER_EPOCH = auto()
    SIMPLE_MOVING_AVERAGE_EPOCH_WINDOW = auto()
