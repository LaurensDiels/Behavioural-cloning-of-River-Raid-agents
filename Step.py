from State import State


class Step:
    """Triples of (action, reward, state), where the action led to the observed reward and state."""

    def __init__(self, action: int, reward: float, state: State):
        self.action = action
        self.reward = reward
        self.state = state
