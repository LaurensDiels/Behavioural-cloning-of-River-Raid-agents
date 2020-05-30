from __future__ import annotations
from typing import Dict, List, Tuple, Iterable
from math import inf, sqrt, log2

import matplotlib.pyplot as plt
import numpy


class EmpiricalDistribution:

    def __init__(self):
        self.observed_counts: Dict = {}
        self.nbObservations = 0

    def observe(self, x):
        self.nbObservations += 1
        if x in self.observed_counts:
            self.observed_counts[x] += 1
        else:
            self.observed_counts[x] = 1

    def observe_list(self, xs: Iterable):
        for x in xs:
            self.observe(x)

    def get_observed_values(self) -> List:
        return list(self.observed_counts.keys())

    def get_count_of_value(self, x) -> int:
        if x in self.get_observed_values():
            return self.observed_counts[x]
        else:
            return 0

    def get_total_nb_of_observations(self) -> int:
        return self.nbObservations

    def get_probability(self, x) -> float:
        return self.get_count_of_value(x) / self.nbObservations

    def KL_divergence(self, d: EmpiricalDistribution) -> float:
        sum = 0.0
        # if x appears in d, but not in self, then Px = 0 and x does not contribute to the sum
        for x in self.get_observed_values():
            if x in d.get_observed_values():
                Px = self.get_probability(x)
                Qx = d.get_probability(x)
                sum += Px * log2(Px / Qx)
            else:
                return inf
        return sum

    def sum_squared_differences(self, d: EmpiricalDistribution) -> float:
        ssd = 0.0
        xs = set().union(self.get_observed_values(), d.get_observed_values())
        for x in xs:
            ssd += (self.get_probability(x) - d.get_probability(x)) ** 2
        return ssd

    def L1_distance(self, d: EmpiricalDistribution) -> float:
        dist = 0.0
        xs = set().union(self.get_observed_values(), d.get_observed_values())
        for x in xs:
            dist += abs(self.get_probability(x) - d.get_probability(x))
        return dist

    def L2_distance(self, d: EmpiricalDistribution) -> float:
        return sqrt(self.sum_squared_differences(d))

    def mean_squared_difference(self, d: EmpiricalDistribution) -> float:
        """Mean over observed values."""
        n = len(set().union(self.get_observed_values(), d.get_observed_values()))
        return self.sum_squared_differences(d) / n

    def root_mean_squared_difference(self, d: EmpiricalDistribution) -> float:
        return sqrt(self.mean_squared_difference(d))

    def __str__(self):
        return self.to_string()

    def to_string(self, xs: List = None, digits_after_decimal_point: int = None) -> str:
        if not xs:
            xs = self.get_observed_values()

        s = ""
        if digits_after_decimal_point and digits_after_decimal_point >= 0:
            format_string = "{0:." + str(digits_after_decimal_point) + "f}"
        else:
            format_string = ""
        for x in xs:
            s += x.__str__() + ": " + format_string.format(self.get_probability(x)) + "; "
        return s

    def get_bar_plot(self, name: str = None, xs: List = None, xs_names: List[str] = None, x_axis_label: str = "",
                     xs_names_rotation: float = 0.0,
                     y_bounds: Tuple[int, int] = None, y_axis_label: str = "Probability",
                     title: str = "") -> plt.Figure:
        return EmpiricalDistribution.get_joint_bar_plot([self], distr_names=[name],
                                                        xs=xs, xs_names=xs_names, x_axis_label=x_axis_label,
                                                        xs_names_rotation=xs_names_rotation,
                                                        y_bounds=y_bounds, y_axis_label=y_axis_label,
                                                        title=title)

    @staticmethod
    def get_joint_bar_plot(eds: List[EmpiricalDistribution], distr_names: List[str] = None,
                           colors: List[Tuple[float, float, float, float]] = None,
                           xs: List = None, xs_names: List[str] = None, xs_names_rotation: float = 0.0,
                           x_axis_label: str = "",
                           y_bounds: Tuple[int, int] = None, y_axis_label: str = "Probability",
                           title: str = "") -> plt.Figure:
        """eds, distr_names, colors, xs, xs_names should have the same name (when set). (The code will run when
        distr_names and/or colors are longer than eds. In that case the last entries will be ignored.)
        If an entry in colors is None, then matplotlib will automatically assign a color."""
        if not distr_names:
            distr_names = [""] * len(eds)
        for name in distr_names:
            if name is None:
                name = ""

        if not colors:
            colors = [None] * len(eds)

        if not xs:
            xs = eds[0].get_observed_values()
            for i in range(1, len(eds)):
                xs = set(xs).union(eds[i].get_observed_values())

        if not xs_names:
            xs_names = [x.__str__() for x in xs]

        fig, ax = plt.subplots()
        bar_width = 1.0 / (len(eds) + 1)  # + 1 for spacing
        for i in range(len(eds)):
            x_locs = numpy.arange(len(xs)) + bar_width * (i + 0.5)  # + 0.5 for spacing between values
            ys = [eds[i].get_probability(x) for x in xs]
            ax.bar(x_locs, ys, width=bar_width, label=distr_names[i], color=colors[i])

        ax.set_xlabel(x_axis_label)
        x_tick_locs = numpy.arange(len(xs)) + bar_width * len(eds) / 2  # in the middle
        ax.set_xticks(x_tick_locs)
        ax.set_xticklabels(xs_names, rotation=xs_names_rotation)

        if y_bounds:
            ax.set_ybound(y_bounds)
        ax.set_ylabel(y_axis_label)

        ax.set_title(title)
        ax.legend()

        return fig


class DiscreteUniformDistribution(EmpiricalDistribution):

    def __init__(self, values: List):
        super().__init__()
        for value in values:
            self.observe(value)




