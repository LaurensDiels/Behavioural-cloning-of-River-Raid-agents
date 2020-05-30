from abc import ABC
import time


class AVisualizationHandler(ABC):

    def __init__(self, environment, do_visualize, do_sleep, FPS):
        self.environment = environment
        self.do_visualize = do_visualize
        self.do_sleep = do_sleep
        if FPS is None:
            self.SPF = None
        else:
            self.SPF = 1.0 / FPS

        self.start_time = time.time()

    def set_start(self):
        self.start_time = time.time()

    def handle_visualization(self):
        if not self.do_visualize:
            return

        self.environment.render('human')

        if not self.do_sleep:
            return

        elapsed_time = time.time() - self.start_time
        if elapsed_time < self.SPF:
            sleep(self.SPF - elapsed_time)  # For some reason time.sleep() no longer works for low durations.


class NoVisualizer(AVisualizationHandler):
    def __init__(self):
        super().__init__(None, False, None, None)


class NoSleepVisualizer(AVisualizationHandler):
    def __init__(self, environment):
        super().__init__(environment, True, False, None)


class MaxFPSVisualizer(AVisualizationHandler):
    def __init__(self, environment, FPS):
        super().__init__(environment, True, True, FPS)


def sleep(duration_in_seconds: float):  # Note: CPU intensive
    end = time.perf_counter_ns() + round(1e9 * duration_in_seconds)
    while time.perf_counter_ns() < end:
        pass
