from PIL import Image
import numpy

from GlobalSettings import *


def preprocess_frame(frame, new_size=(RL_PREPROCESS_WIDTH, RL_PREPROCESS_HEIGHT), grayscale=RL_PREPROCESS_GRAYSCALE):
    # Based on _preprocess_frame in video.py of keras_gym.
    img = Image.fromarray(frame)
    if grayscale:
        img = img.convert('L')
    img = img.resize(new_size)
    return numpy.array(img)


def repeat_n_times(np_array: numpy.ndarray, n: int):
    x = np_array
    x = x.reshape(x.shape + (1,))
    c = x
    for _ in range(n - 1):
        c = numpy.concatenate((c, x), axis=len(c.shape)-1)
    return c