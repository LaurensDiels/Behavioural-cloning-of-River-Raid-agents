from typing import List

from Step import Step
from State import State


class Episode:

    def __init__(self, initial_state: State):
        self.steps = [Step(None, None, initial_state)]

    def add_step(self, action: int, reward: float, new_state: State):
        self.steps.append(Step(action, reward, new_state))

    def get_subepisode(self, length: int):
        return Episode.create_episode(self.steps[0:length])

    def finished(self) -> bool:
        return self.steps[-1].state.is_terminal

    def get_score(self) -> float:
        if len(self.steps) <= 1:
            return 0.0  # In the initial step there was no reward (None).
        else:
            return sum(map(lambda step: step.reward, self.steps[1:]))

    def current_length(self) -> int:
        return len(self.steps)

    def latest_action(self) -> int:
        return self.steps[-1].action

    def latest_reward(self) -> float:
        return self.steps[-1].reward

    def current_state(self) -> State:
        return self.steps[-1].state

    def previous_state(self) -> State:
        """Returns None if there is no previous state."""
        if len(self.steps) <= 1:
            return None
        else:
            return self.steps[-2].state

    @classmethod
    def create_episode(cls, steps: List[Step]):
        self = cls.__new__(cls)
        self.steps = steps
        return self
