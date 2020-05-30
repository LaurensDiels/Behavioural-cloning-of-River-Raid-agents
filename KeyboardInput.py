import keyboard
from enum import Enum, auto
from AtariActions import *
from KeyboardScanCodes import *


class InputTypes(Enum):
    SEPARATE = auto()  # One key for one action
    COMBINED = auto()  # Keys can be combined

# For InputTypes.SEPARATE:
#
# Controls: centered around s and l:
# around s: movement in direction (left, right, speed up, slow down),
# around l: shoot forward + move in direction;
# spacebar: shoot.
# Actions don't combine: to move to the right while firing
# don't press d and space at the same time,
# use semicolon (m for azerty).
# Only the 'first' input will be used.
#
#
# For InputTypes.COMBINED:
#
# More standard setup with wasd (or arrow keys) to move and space to fire.
# Keys can be combined: e.g. s + d + space -> DOWNRIGHTFIRE


def get_input_action(input_type: InputTypes):
    if input_type == InputTypes.SEPARATE:
        if keyboard.is_pressed(KEY_Q):
            return A_UPLEFT
        if keyboard.is_pressed(KEY_W):
            return A_UP
        if keyboard.is_pressed(KEY_E):
            return A_UPRIGHT
        if keyboard.is_pressed(KEY_A):
            return A_LEFT
        if keyboard.is_pressed(KEY_D):
            return A_RIGHT
        if keyboard.is_pressed(KEY_Z):
            return A_DOWNLEFT
        if keyboard.is_pressed(KEY_X):
            return A_DOWN
        if keyboard.is_pressed(KEY_C):
            return A_DOWNRIGHT

        if keyboard.is_pressed(KEY_SPACEBAR):
            return A_FIRE

        if keyboard.is_pressed(KEY_I):
            return A_UPLEFTFIRE
        if keyboard.is_pressed(KEY_O):
            return A_UPFIRE
        if keyboard.is_pressed(KEY_P):
            return A_UPRIGHTFIRE
        if keyboard.is_pressed(KEY_K):
            return A_LEFTFIRE
        if keyboard.is_pressed(KEY_SEMICOLON):
            return A_RIGHTFIRE
        if keyboard.is_pressed(KEY_COMMA):
            return A_DOWNLEFTFIRE
        if keyboard.is_pressed(KEY_FULLSTOP):
            return A_DOWNFIRE
        if keyboard.is_pressed(KEY_FORWARDSLASH):
            return A_DOWNRIGHTFIRE

    elif input_type == InputTypes.COMBINED:
        up = keyboard.is_pressed(KEY_W) or keyboard.is_pressed(KEY_ARROWUP)
        left = keyboard.is_pressed(KEY_A) or keyboard.is_pressed(KEY_ARROWLEFT)
        right = keyboard.is_pressed(KEY_D) or keyboard.is_pressed(KEY_ARROWRIGHT)
        down = keyboard.is_pressed(KEY_S) or keyboard.is_pressed(KEY_ARROWDOWN)
        fire = keyboard.is_pressed(KEY_SPACEBAR)

        # Canceling actions
        if left and right:
            left = False
            right = False
        if up and down:
            up = False
            down = False

        if not fire:
            if up and left:
                return A_UPLEFT
            if up and not left and not right:
                return A_UP
            if up and right:
                return A_UPRIGHT
            if not up and not down and left:
                return A_LEFT
            if not up and not down and right:
                return A_RIGHT
            if down and left:
                return A_DOWNLEFT
            if down and not left and not right:
                return A_DOWN
            if down and right:
                return A_DOWNRIGHT
            return A_NOOP
        else:
            if up and left:
                return A_UPLEFTFIRE
            if up and not left and not right:
                return A_UPFIRE
            if up and right:
                return A_UPRIGHTFIRE
            if not up and not down and left:
                return A_LEFTFIRE
            if not up and not down and right:
                return A_RIGHTFIRE
            if down and left:
                return A_DOWNLEFTFIRE
            if down and not left and not right:
                return A_DOWNFIRE
            if down and right:
                return A_DOWNRIGHTFIRE
            return A_FIRE

    return A_NOOP


def input_quit() -> bool:
    return keyboard.is_pressed('esc')
