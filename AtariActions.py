# Atari actions
A_FIRST = 0

A_NOOP = 0
A_FIRE = 1
A_UP = 2
A_RIGHT = 3
A_LEFT = 4
A_DOWN = 5
A_UPRIGHT = 6
A_UPLEFT = 7
A_DOWNRIGHT = 8
A_DOWNLEFT = 9
A_UPFIRE = 10
A_RIGHTFIRE = 11
A_LEFTFIRE = 12
A_DOWNFIRE = 13
A_UPRIGHTFIRE = 14
A_UPLEFTFIRE = 15
A_DOWNRIGHTFIRE = 16
A_DOWNLEFTFIRE = 17

A_LAST = 17

NB_ACTIONS = A_LAST - A_FIRST + 1


def atari_action_to_name(action: int) -> str:
    if action == 0: return "No action"
    if action == 1: return "Fire"
    if action == 2: return "Up"
    if action == 3: return "Right"
    if action == 4: return "Left"
    if action == 5: return "Down"
    if action == 6: return "Up + Right"
    if action == 7: return "Up + Left"
    if action == 8: return "Down + Right"
    if action == 9: return "Down + Left"
    if action == 10: return "Up + Fire"
    if action == 11: return "Right + Fire"
    if action == 12: return "Left + Fire"
    if action == 13: return "Down + Fire"
    if action == 14: return "Up + Right + Fire"
    if action == 15: return "Up + Left + Fire"
    if action == 16: return "Down + Right + Fire"
    if action == 17: return "Down + Left + Fire"


def atari_action_to_short_name(action: int) -> str:
    if action == 0: return "None"
    if action == 1: return "Fire"
    if action == 2: return "Up"
    if action == 3: return "Right"
    if action == 4: return "Left"
    if action == 5: return "Down"
    if action == 6: return "UR"
    if action == 7: return "UL"
    if action == 8: return "DR"
    if action == 9: return "DL"
    if action == 10: return "UF"
    if action == 11: return "RF"
    if action == 12: return "LF"
    if action == 13: return "DF"
    if action == 14: return "URF"
    if action == 15: return "ULF"
    if action == 16: return "DRF"
    if action == 17: return "DLF"
