import math
from typing import List


def running_average(seq: List[float]) -> List[float]:
    """Also known as cumulative moving average.
    Every smoothed entry will be the average of all previous unsmoothed ones."""
    ra = [seq[0]]
    for i in range(1, len(seq)):
        ra.append((i * ra[i - 1] + seq[i]) / (i + 1))
    return ra


def undo_running_average(ra: List[float]) -> List[float]:
    """Satisfies undo_running_average(running_average(seq)) == seq."""
    seq = [ra[0]] * len(ra)
    for i in range(1, len(ra)):
        seq[i] = (i + 1) * ra[i] - i * ra[i - 1]
    return seq


def resetting_running_average(seq: List[float], reset_after_nb: int) -> List[float]:
    """A running average which will reset after every reset_after_nb point. This corresponds to splitting the
    sequence in blocks of size reset_after_nb, computing the running average of every block separately, and finally
    concatenating the results.
    E.g. for seq=[1,2,3,4,5] and reset_after_nb=2, the output is [1,1.5,3,3.5,5]."""
    return resetting_running_average_irregular_split(seq, [reset_after_nb] * math.floor(len(seq)/reset_after_nb))


def resetting_running_average_irregular_split(seq: List[float], reset_after_nbs: List[int]) -> List[float]:
    """Similar to resetting_running_average, but now we might first reset after e.g. 3 points, the next time after 10
    points, then after 2 points, etc. In other words, the blocks in which we split the sequence need no longer have
    equal size (to the extent possible). These sizes form reset_after_nbs."""
    rra = []
    steps_since_last_reset = 0
    reset_nb = 0
    for i in range(len(seq)):
        if steps_since_last_reset == 0:
            rra.append(seq[i])
        else:
            rra.append((steps_since_last_reset * rra[i - 1] + seq[i]) / (steps_since_last_reset + 1))
        steps_since_last_reset += 1
        if reset_nb < len(reset_after_nbs) and steps_since_last_reset >= reset_after_nbs[reset_nb]:
            # If we have gone through
            steps_since_last_reset = 0
            reset_nb += 1
    return rra


def exponential_moving_average(seq: List[float], decay_factor: float) -> List[float]:
    """Every next smoothed entry will be a convex combination of the current smoothed entry (diminished by
    decay_factor (in [0, 1]) and the next unsmoothed entry (diminished by (1 - decay_factor)).
    If decay_factor is close to 0 the output sequence will be very similar to the input sequence, with little smoothing.
    If it is very close to 1 the output sequence will be very smoothed, possibly at the expense of actually tracking
    the input sequence."""
    ema = [seq[0]]
    for i in range(1, len(seq)):
        ema.append(decay_factor * ema[i - 1] + (1 - decay_factor) * seq[i])
    return ema


def simple_moving_average_backwards(seq: List[float], window_size: int) -> List[float]:
    """In general every entry i in the smoothed output will be the average of the windows_size previous entries
    i, i - 1, ..., i - window_size + 1 unsmoothed entries in the input list.
    For the first window_size output entries this is not possible, and we will use a running average there (i.e. every
    entry will be the average of all (< window_size) previous unsmoothed entries)."""
    sma_b = [seq[0]]
    # running average (incomplete simple moving average)
    for i in range(1, window_size):
        sma_b.append((i * sma_b[i - 1] + seq[i]) / (i + 1))
    # complete simple moving average
    for i in range(window_size, len(seq)):
        sma_b.append(sma_b[i - 1] + (seq[i] - seq[i - window_size]) / window_size)
    return sma_b


def simple_moving_average_central(seq: List[float], radius: int, shrink_radius_at_boundaries: bool = False) \
        -> List[float]:
    """The smoothed entries will be the averages of the current unsmoothed entries and the next and previous radius
    ones. At the boundaries this is not possible, and if shrink_radius_at_boundaries is True, we reduce the radius to
    fit. So the first output entry will simply be the input entry, the second the average of the first three inputs,
    etc. Until we can just use the average of the first 2 * inputs + 1 entries. At the end of the list the same happens.
    If shrink_radius_at_boundaries is set to False (as is the case by default), we will always replace an entry in the
    unsmoothed sequence by taking the average over all _existing_ entries within the radius. For example the first entry
    will be replaced by the average of this entry and the next radius ones. The second by the average of the first,
    second and next radius ones, etc."""
    sma_c = [0.0] * len(seq)

    if 2 * radius + 1 > len(seq):
        # If all of the sequence is considered boundary.
        radius = (len(seq) - 1) // 2
        # This makes no difference to the output.

    if shrink_radius_at_boundaries:
        # Left boundary and right boundary:
        # use the central simple moving average approach with maximal radius (smaller than or equal to the global
        # radius)
        for i in range(radius + 1):
            sma_c[i] = average(seq[0:2*i+1])
            sma_c[-1 - i] = average(seq[-1-2*i:])
    else:
        # Left boundary and right boundary:
        # take the average of all existing entries in the radius. Since we average over an increasing set consisting of
        # the same elements plus a new one, we can efficiently reuse previous computations, similar to what we did for
        # the running average.

        # Left boundary
        sma_c[0] = average(seq[0:radius + 1])  # average of {seq[0], ..., seq[radius]}
        for i in range(1, radius + 1):
            sma_c[i] = ((sma_c[i - 1] * (radius + i)) + seq[i + radius]) / (radius + i + 1)
            # For j <= radius, sma_c[j] is the average over radius + j + 1 elements, namely 0 to j + radius

        # Right boundary
        sma_c[-1] = average(seq[-radius - 1:])  # average of the last radius + 1 entries of seq
        for i in range(1, radius + 1):
            sma_c[-i - 1] = ((sma_c[-i] * (radius + i)) + seq[-i - radius - 1]) / (radius + i + 1)

    # Complete simple moving average
    for i in range(radius + 1, len(seq)-1 - radius):
        sma_c[i] = sma_c[i - 1] + (seq[i + radius] - seq[i - radius - 1]) / (2 * radius + 1)
    return sma_c


def average(lst: List[float]) -> float:  # (or use numpy.mean)
    return sum(lst) / len(lst)
