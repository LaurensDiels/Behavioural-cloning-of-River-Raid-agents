import math
import pickle
import compress_pickle
from typing import Dict, Tuple, List

import gym
import matplotlib.pyplot as plt
import numpy

import AtariActions
from AtariActions import atari_action_to_short_name, NB_ACTIONS
from DataLoader import EpisodeDataLoader
from Episode import Episode
from InputHandler import AInputHandler, KerasModelInputHandler, KerasGymModelInputHandler, RandomInputHandler
from EmpiricalDistribution import EmpiricalDistribution, DiscreteUniformDistribution
from KS2D import ks2d2s
from State import State

from GlobalSettings import *


def load_RL_BC_agents() -> Dict[str, AInputHandler]:
    """i.e. RL and the clones of varying number of episodes"""
    agents_ih = {"RL": KerasGymModelInputHandler(FIXED_RL_PI_PREDICT_MODEL_WEIGHTS_PATH)}
    for i in range(len(RL_BC_VARYING_NB_EPISODES_NBS)):
        nb = RL_BC_VARYING_NB_EPISODES_NBS[i]
        agents_ih["BC {} episodes".format(nb)] = \
            KerasModelInputHandler(RL_BC_VARYING_NB_EPISODES_MODEL_PATHS[i],
                                   CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_NB_FRAMES_TO_STACK, True)
    return agents_ih


def load_random_BC_agents() -> Dict[str, AInputHandler]:
    """i.e. the random agent and its BC"""
    return {"Random": RandomInputHandler(),
            "Random BC": KerasModelInputHandler(RANDOM_BC_MODEL_PATH,
                                                CUSTOM_A_TUNED_HYPERPARAMETERS_RANDOM_BC_NB_FRAMES_TO_STACK, True)}


def load_human_BC_agents() -> Dict[str, AInputHandler]:
    """i.e. the BCs, not the human agent itself (which we can't load for obvious reasons)"""
    return {
            "BC Scratch": KerasModelInputHandler(HUMAN_BC_CUSTOM_A_MODEL_SCRATCH_MODEL_PATH,
                                                 CUSTOM_A_TUNED_HYPERPARAMETERS_HUMAN_BC_NB_FRAMES_TO_STACK, True),
            "BC TL Init": KerasModelInputHandler(HUMAN_BC_CUSTOM_A_MODEL_TL_INIT_MODEL_PATH,
                                                 CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_NB_FRAMES_TO_STACK, True),
            "BC TL New top": KerasModelInputHandler(HUMAN_BC_CUSTOM_A_MODEL_TL_FREEZE_MODEL_PATH,
                                                    CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_NB_FRAMES_TO_STACK, True),
            "BC TL New top Finetune": KerasModelInputHandler(HUMAN_BC_CUSTOM_A_MODEL_TL_FINETUNE_MODEL_PATH,
                                                             CUSTOM_A_TUNED_HYPERPARAMETERS_RL_BC_NB_FRAMES_TO_STACK,
                                                            True)
            # (Recall that for the transfer learning to work, we needed to use the same number of stacked frames as
            #  in the RL case.)
    }


def compare_distributions(agents_ih: Dict[str, AInputHandler], base_agent_name: str,
                          save: bool, save_path_without_extension: str,
                          verbose: bool,
                          max_agents_per_plot: int = 5):
    distributions = gather_distributions(agents_ih, verbose, base_agent_name)
    if save:
        save_distributions(distributions, save_path_without_extension)

    plot_split = None
    if max_agents_per_plot is not None and max_agents_per_plot >= 2:
        plot_split = create_plot_split(list(agents_ih.keys()), base_agent_name, max_agents_per_plot, True)
    analyze_distributions(distributions, base_agent_name, save, save_path_without_extension, plot_split)


def gather_distributions(agents_ih: Dict[str, AInputHandler], verbose: bool, base_agent_name: str = None) -> Dict:
    """The returned dictionary has two entries:
    "global" : dictionary of EmpiricalDistributions of the different agents
    "list_of_local" : list (per state) of dictionary of EmpiricalDistributions."""
    HUMAN_EPISODE_NUMBER = 5
    NB_OBSERVATIONS_PER_STATE = 100
    STEP = 10

    if MODEL_DISTRIBUTION_COMPARISON_FOLDER and not os.path.exists(MODEL_DISTRIBUTION_COMPARISON_FOLDER):
        os.makedirs(MODEL_DISTRIBUTION_COMPARISON_FOLDER)

    edl = EpisodeDataLoader("Data//HumanBC//Episodes", EPISODE_PREFIX, HUMAN_EPISODE_NUMBER)

    uniform_distribution = DiscreteUniformDistribution(list(range(NB_ACTIONS)))

    global_distributions = {agent: EmpiricalDistribution() for agent in agents_ih}
    local_distributions_list = []

    for i in range((edl.episode.current_length() - 1) // STEP):  # -1 as we start from subepisode of length 1
        if verbose:
            print("{} / {}".format(i + 1, edl.episode.current_length() // STEP))
        partial_episode = edl.episode.get_subepisode(STEP * i + 1)  # start from length 1

        local_distributions = {agent: EmpiricalDistribution() for agent in agents_ih}

        for _ in range(NB_OBSERVATIONS_PER_STATE):
            for agent_name in agents_ih:
                a = agents_ih[agent_name].get_action(partial_episode)
                global_distributions[agent_name].observe(a)
                local_distributions[agent_name].observe(a)

        local_distributions_list.append(local_distributions)

        if verbose and base_agent_name is not None:
            for agent_name in agents_ih:
                print("{} vs {}: {:.3f}".format(base_agent_name, agent_name,
                                                local_distributions[base_agent_name].L1_distance(
                                                    local_distributions[agent_name])))
            print("{} vs uniform: {:.3f}".format(base_agent_name,
                                                 local_distributions[base_agent_name].L1_distance(uniform_distribution))
                  )

    distributions = {"global": global_distributions, "list_of_local": local_distributions_list}
    return distributions


def save_distributions(distributions: Dict, save_path_without_extension: str):
    folder = os.path.dirname(save_path_without_extension)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(save_path_without_extension + ".pcl", "wb") as file:
        pickle.dump(distributions, file)


def analyze_distributions(distributions: Dict, base_agent_name: str,
                          save: bool, save_path_without_extension: str, plot_split: List[List[str]] = None):
    """If plot_split is None, the plot will not be split. Otherwise it should be created using create_plot_split.
    If the dictionary distributions does not contain the key 'list_of_local', no analysis of local distributions will
    be performed."""

    global_distributions = distributions["global"]
    agent_names = list(global_distributions.keys())
    if plot_split is None:
        plot_split = [agent_names]
    plot_split_colors = assign_plot_split_colors(plot_split, base_agent_name)

    # Qualitatively using a plot
    for i in range(len(plot_split)):
        plot_split_agent_names = plot_split[i]
        fig = EmpiricalDistribution.get_joint_bar_plot([global_distributions[agent_name]
                                                        for agent_name in plot_split_agent_names],
                                                       distr_names=plot_split_agent_names,
                                                       colors=plot_split_colors[i],
                                                       xs=list(range(NB_ACTIONS)),
                                                       xs_names=[atari_action_to_short_name(a)
                                                                 for a in range(NB_ACTIONS)],
                                                       xs_names_rotation=-45,
                                                       x_axis_label="Actions",
                                                       # y_bounds=(0, 1),
                                                       y_axis_label="Probability",
                                                       title="Action distributions"
                                                       )
        if save:
            if len(plot_split) > 1:
                fig.savefig(save_path_without_extension + "_part{}.svg".format(i))
            else:
                fig.savefig(save_path_without_extension + ".svg")
        else:
            plt.show()

    # Quantitatively
    #  Global
    for agent_name in agent_names:
        print("L1 distance between the global distributions of {} and {}: {}".format(
            base_agent_name, agent_name,
            global_distributions[base_agent_name].L1_distance(global_distributions[agent_name])))
    uniform_distribution = DiscreteUniformDistribution(list(range(NB_ACTIONS)))
    print("L1 distance between the global distributions of {} and a uniform distribution: {}".format(
          base_agent_name, global_distributions[base_agent_name].L1_distance(uniform_distribution)))

    #  Local
    if "list_of_local" in distributions:
        local_distributions_list = distributions["list_of_local"]
        for agent_name in agent_names:
            L1_distances = []
            for local_distributions in local_distributions_list:
                L1_distances.append(local_distributions[base_agent_name].L1_distance(local_distributions[agent_name]))
            print("average conditional L1 distance between the local distributions of {} and {}: {}".format(
                base_agent_name, agent_name, numpy.mean(L1_distances)))
        #   Uniform
        uniform_distribution = DiscreteUniformDistribution(list(range(NB_ACTIONS)))
        L1_distances = []
        for local_distributions in local_distributions_list:
            L1_distances.append(local_distributions[base_agent_name].L1_distance(uniform_distribution))
        print("average conditional L1 distance between the local distributions of {} and a uniform distribution: {}".format(
            base_agent_name, numpy.mean(L1_distances)))


def create_plot_split(agent_names: List[str], base_agent_name: str, max_in_single_plot: int,
                      place_base_agent_in_middle: bool) \
        -> List[List[str]]:
    """Splits the agent_names into balanced lists of length max_in_single_plot all of which contain base_agent_name.
    If place_base_agent_in_middle is set to True, base_agent_name will be the first entry in all these balanced lists.
    Otherwise we will it near the middle if there is an odd number of agents in a plot, or just left of the middle
    otherwise.
    E.g. with agent_names=['Base', 'BC1', 'BC2', 'BC3'], base_agent_name='Base', max_in_single_plot=3, and
    place_base_agent_in_middle=False, the output
    will be [['Base', 'BC1', 'BC2'], ['Base', 'BC3']]. If place_base_agent_in_middle would be set to True the first
    list would change to ['BC1', 'Base', 'BC2'], but the second would remain the same.

    If base_agent_name or max_in_single_plot is None, the plot will not be split. In this case the output is a list
    with as single element agent_names. The same output will be returned if there are at most two agents in agent_names.
    """

    if len(agent_names) <= 2 or base_agent_name is None or max_in_single_plot is None:
        return [agent_names]

    """
    Let n be the number of non-base agents, and M be the maximum number of agents in a single plot. If we use k
    plots, then the base agent will appear k times in total, giving k + n agents in total. Since we use k plots,
    at most ceil((k + n)/k) agents (assuming a somewhat equal distribution) will on average appear in each plot. 
    This must be bounded by M,  yielding k >= n/(M-1). Since we want to use as few plots as possible, 
    k = ceil(n/(M-1)).

    If we want to distribute the agents as evenly as possible, we can do that by using either f = floor((k + n)/k)
    or c = ceil((k + n)/k) agents in a plot. If we would consistently use f, we would only use kf of the k + n
    agents, meaning we ignored (0 <=) k(1 - f) + n (< k) of them. We can simply add them one by one to the plots.
    """
    nb_non_base_agents = len(agent_names) - 1  # n; data includes the base agent
    nb_plots = math.ceil(nb_non_base_agents / (max_in_single_plot - 1))  # nb_plots = k, max_in_single_plot = m
    min_in_plot = 1 + math.floor(nb_non_base_agents / nb_plots)
    missed = nb_plots * (1 - min_in_plot) + nb_non_base_agents

    plot_split = []
    base_agent_min_location = (min_in_plot - 1) // 2 if place_base_agent_in_middle else 0
    # If min_in_plot is even [e.g. 4] and the current (first) plot_list will contain one more entry [5] than the
    # minimum because we cannot perfectly balance the plots, then the location of the base agent will shift one to the
    # right [to 2 instead of 1].
    base_agent_location = base_agent_min_location + 1 if place_base_agent_in_middle and missed >= 1 \
                                                         and min_in_plot % 2 == 0 \
        else base_agent_min_location
    plot_list = [] if base_agent_location != 0 else [base_agent_name]

    for i in range(len(agent_names)):
        agent_name = agent_names[i]
        if agent_name == base_agent_name:
            continue

        plot_list.append(agent_name)
        if base_agent_location == len(plot_list):
            plot_list.append(base_agent_name)  # so that plot_list[base_agent_location] == base_agent_name

        plot_list_is_not_full = len(plot_list) < min_in_plot or (len(plot_list) == min_in_plot and missed >= 1)
        if not plot_list_is_not_full:
            if len(plot_list) > min_in_plot:
                missed -= 1
                base_agent_location = base_agent_min_location + 1 if place_base_agent_in_middle and missed >= 1 \
                                                                     and min_in_plot % 2 == 0 \
                    else base_agent_min_location
            plot_split.append(plot_list)
            plot_list = [] if base_agent_location != 0 else [base_agent_name]
    if len(plot_list) > 1:  # if it does not just contain the base agent
        plot_split.append(plot_list)

    return plot_split


def assign_plot_split_colors(plot_split: List[List[str]], base_agent_name: str = None) \
        -> List[List[Tuple[float, float, float, float]]]:
    """If base_agent_name is set to None, we will try to infer it from the plot_split, namely as being the (normally)
    only agent appearing in both the zeroth and first plots in the split. If there is only one plot, we will take the
    first agent there."""
    max_agents_in_plot = max([len(plot_split_agent_names) for plot_split_agent_names in plot_split])
    if max_agents_in_plot <= 10:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # default colors
    else:  # Note: having more than 10 agents in the same plot is strongly not recommended.
        c_map = plt.cm.get_cmap('gist_rainbow', max_agents_in_plot)  # can take as many as we want
        colors = [c_map(i) for i in range(max_agents_in_plot)]

    if base_agent_name is None:
        if len(plot_split) >= 2:
            # The base agent should be the only one common to two split plots.
            base_agent_name = set(plot_split[0]).intersection(set(plot_split[1])).pop()
        else:
            # In this case this really does not matter as the color will not be reused.
            base_agent_name = plot_split[0][0]

    plot_split_colors = []
    for plot_split_agent_names in plot_split:
        j = 1  # color index, 0 is reserved for the base agent
        pl_colors = []
        for agent_name in plot_split_agent_names:
            if agent_name == base_agent_name:
                pl_colors.append(colors[0])
            else:
                pl_colors.append(colors[j])
                j += 1
        plot_split_colors.append(pl_colors)

    return plot_split_colors


def compare_score_and_time_alive(agents_ih: Dict[str, AInputHandler], base_agent_name: str,
                                 save: bool, save_path_without_extension: str, verbose: bool,
                                 max_agents_per_plot: int = 3):
    data = gather_score_and_time_alive_data(agents_ih, verbose)
    if save:
        save_score_and_time_alive_data(data, save_path_without_extension)

    plot_split = None
    if max_agents_per_plot is not None and max_agents_per_plot >= 2:
        plot_split = create_plot_split(list(data.keys()), base_agent_name, max_agents_per_plot, False)
    analyze_score_and_time_alive_data(data, base_agent_name, save, save_path_without_extension, plot_split)


def gather_score_and_time_alive_data(agents_ih: Dict[str, AInputHandler], verbose: bool) \
        -> Dict[str, List[Tuple[int, float]]]:
    """The agents will be forced to make a noop move at the start (namely down) if otherwise they would refuse to
    properly start the game."""
    NB_EPISODES = 100

    if MODEL_SCORES_TIME_ALIVE_COMPARISON_FOLDER and not os.path.exists(MODEL_SCORES_TIME_ALIVE_COMPARISON_FOLDER):
        os.makedirs(MODEL_SCORES_TIME_ALIVE_COMPARISON_FOLDER)

    data = {}  # will be of the form agent_name: list_of_time_score_pairs
    for agent_name in agents_ih:
        agent = agents_ih[agent_name]
        data[agent_name] = []
        if verbose:
            print("{}:::::::::::::::::::::::::".format(agent_name))

        for i in range(NB_EPISODES):
            if verbose:
                print("Episode {}".format(i))
            env = gym.make("Riverraid-v0")
            initial_observation = env.reset()
            episode = Episode(State(initial_observation, False))
            started = False
            while not (episode.finished()):
                action = agent.get_action(episode)
                if not started and action == AtariActions.A_NOOP:
                    action = AtariActions.A_DOWN
                observation, reward, done, _ = env.step(action)
                started = not numpy.array_equal(observation, episode.current_state().screen)
                episode.add_step(action, reward, State(observation, done))

            env.close()
            data[agent_name].append((episode.current_length(), episode.get_score()))

    return data


def save_score_and_time_alive_data(data: Dict[str, List[Tuple[int, float]]], save_path_without_extension: str):
    folder = os.path.dirname(save_path_without_extension)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(save_path_without_extension + ".pcl", "wb") as file:
        pickle.dump(data, file)


def analyze_score_and_time_alive_data(data: Dict[str, List[Tuple[int, float]]], base_agent_name: str,
                                      save: bool, save_path_without_extension: str,
                                      plot_split: List[List[str]] = None):
    """If plot_split is None, we won't split the plot. Otherwise it should be created using create_plot_split."""
    if plot_split is None:
        plot_split = [list(data.keys())]  # don't split the plot
    plot_split_colors = assign_plot_split_colors(plot_split, base_agent_name)

    # Plot (Qualitative analysis)
    for i in range(len(plot_split)):
        plot_split_agent_names = plot_split[i]

        fig, ax = plt.subplots()
        ax.set_xlabel("Survived frames")
        ax.set_ylabel("Score")
        ax.set_title("Score vs time survived")

        for j in range(len(plot_split_agent_names)):
            agent_name = plot_split_agent_names[j]
            agent_data = numpy.array(data[agent_name])
            ax.scatter(agent_data[:, 0], agent_data[:, 1], alpha=0.5, label=agent_name, color=plot_split_colors[i][j])

        ax.legend()
        if save:
            if len(plot_split) > 1:
                fig.savefig(save_path_without_extension + "_part{}.svg".format(i))
            else:
                fig.savefig(save_path_without_extension + ".svg")
        else:
            plt.show()

    # Quantitative analysis
    # KL divergence between fitted Gaussians
    m_base, S_base = fit_Gaussian(data[base_agent_name])
    for agent_name in data:
        # Don't exclude the base agent as a sanity check (the KL divergence to itself should be 0).
        m, S = fit_Gaussian(data[agent_name])
        KL_div = Gaussian_KL_divergence(m_base, S_base, m, S)
        print("KL divergence between fitted Gaussians for {} and {}: {}".format(base_agent_name, agent_name, KL_div))

    # 2D Kolmogorov-Smirnov two-sample test for comparing the (non-parametrized) distributions
    for agent_name in data:
        _, p = ks2d2s(numpy.array(data[base_agent_name]), numpy.array(data[agent_name]))
        print("KS probability of {} and {} data coming from the same distribution: {}".format(base_agent_name,
                                                                                              agent_name, p))


def fit_Gaussian(points: List[Tuple[int, float]]) -> (numpy.ndarray, numpy.matrix):
    np_points = numpy.asarray(points)  # The 0-th component refers to the example number, the 1st to the time or score.
    mean = numpy.mean(np_points, axis=0)
    covariance_matrix = numpy.cov(np_points, rowvar=False, bias=True)  # maximum likelihood estimation
    return mean, covariance_matrix


def Gaussian_KL_divergence(mean_1, cov_1, mean_2, cov_2):
    dim = len(mean_1)
    inv_cov_2 = numpy.linalg.inv(cov_2)
    # (According to the formula on Wikipedia:)
    KL_nats = 0.5 * (numpy.trace(numpy.matmul(inv_cov_2, cov_1))
                     + numpy.matmul(numpy.matmul(numpy.transpose(mean_2 - mean_1), inv_cov_2), mean_2 - mean_1)
                     - dim + numpy.log(numpy.linalg.det(cov_2) / numpy.linalg.det(cov_1)))
    KL_bits = KL_nats / numpy.log(2)

    return KL_bits


def compare_distributions_human(save: bool, save_path_without_extension: str,
                                verbose: bool,
                                max_agents_per_plot: int = 5):
    global_distributions = gather_global_distributions_human(load_human_BC_agents(), verbose)
    distributions = {"global": global_distributions}

    # From this point onwards do the same as in compare_distributions (with base_agent_name="Human"):
    if save:
        save_distributions(distributions, save_path_without_extension)

    plot_split = None
    if max_agents_per_plot is not None and max_agents_per_plot >= 2:
        plot_split = create_plot_split(list(global_distributions.keys()), "Human", max_agents_per_plot, True)
    analyze_distributions(distributions, "Human", save, save_path_without_extension, plot_split)


def gather_global_distributions_human(clones_ih: Dict[str, AInputHandler], verbose: bool) \
        -> Dict[str, EmpiricalDistribution]:
    """We will force episodes to start. Actions before the game actually acknowledges the actions will be ignored in the
    case of the clones."""
    global_distributions = {}

    # Human
    human_empirical_global_distribution = EmpiricalDistribution()
    with open(HUMAN_BC_CUSTOM_A_MODEL_FILE_SPLIT_PATH, "rb") as file:
        split_file_nbs = pickle.load(file)

    nb_episodes = len(split_file_nbs["test"])
    for i in range(len(split_file_nbs["test"])):
        test_file_nb = split_file_nbs["test"][i]
        if verbose:
            print("Collecting global distribution from human test episodes ({}/{})...".format(i + 1, nb_episodes))
        edl = EpisodeDataLoader(HUMAN_EPISODES_FOLDER, EPISODE_PREFIX, test_file_nb)
        human_empirical_global_distribution.observe_list(edl.load_actions())
    global_distributions["Human"] = human_empirical_global_distribution

    # Clones
    for agent_name in clones_ih:
        agent_ih = clones_ih[agent_name]
        clone_empirical_global_distribution = EmpiricalDistribution()
        for i in range(nb_episodes):
            if verbose:
                print("Collecting global distribution from {} episodes ({}/{})...".format(agent_name, i + 1,
                                                                                          nb_episodes))
            env = gym.make("Riverraid-v0")
            initial_observation = env.reset()
            episode = Episode(State(initial_observation, False))

            started = False
            while not episode.finished():
                action = agent_ih.get_action(episode)
                if not started and action == AtariActions.A_NOOP:
                    action = AtariActions.A_DOWN
                if started:
                    clone_empirical_global_distribution.observe(action)
                observation, reward, done, _ = env.step(action)
                started = not numpy.array_equal(observation, episode.current_state().screen)
                episode.add_step(action, reward, State(observation, done))

            env.close()
        global_distributions[agent_name] = clone_empirical_global_distribution

    return global_distributions


def compare_score_and_time_alive_human(save: bool, save_path_without_extension: str, verbose: bool,
                                       max_agents_per_plot: int = 3):
    data = {"Human": gather_human_score_and_time_alive_data()}
    data.update(gather_score_and_time_alive_data(load_human_BC_agents(), verbose))  # for the clones
    if save:
        save_score_and_time_alive_data(data, save_path_without_extension)

    plot_split = None
    if max_agents_per_plot is not None and max_agents_per_plot >= 2:
        plot_split = create_plot_split(list(data.keys()), "Human", max_agents_per_plot, False)
    analyze_score_and_time_alive_data(data, "Human", save, save_path_without_extension, plot_split)


def gather_human_score_and_time_alive_data() -> List[Tuple[int, float]]:
    with open(HUMAN_BC_CUSTOM_A_MODEL_FILE_SPLIT_PATH, "rb") as file:
        split_file_nbs = pickle.load(file)

    human_data = []
    for test_file_nb in split_file_nbs["test"]:
        edl = EpisodeDataLoader(HUMAN_EPISODES_FOLDER, EPISODE_PREFIX, test_file_nb)
        episode = edl.episode
        human_data.append((episode.current_length(), episode.get_score()))
    return human_data


def compare_fixed_frame_variability(save: bool, save_path_without_extension: str, verbose: bool,
                                    frame_nbs: List[int] = None, nb_frames_to_gather_for_computer_agents: int = 100):
    if frame_nbs is None:
        frame_nbs = [60]
    frames_dict = gather_fixed_frames(verbose, frame_nbs, nb_frames_to_gather_for_computer_agents)
    if save:
        save_fixed_frames_dict(frames_dict, save_path_without_extension)
    analyze_fixed_frames_dict(frames_dict, save, save_path_without_extension)


def gather_fixed_frames(verbose: bool, frame_nbs: List[int], nb_frames_to_gather_for_computer_agents: int) \
        -> Dict[str, Dict[int, List[numpy.ndarray]]]:
    return {
        "human": gather_fixed_frame_nb_human(verbose, frame_nbs),
        "RL": gather_fixed_frame_nb_input_handler(KerasGymModelInputHandler(FIXED_RL_PI_PREDICT_MODEL_WEIGHTS_PATH),
                                                  "RL",
                                                  verbose,
                                                  frame_nbs,
                                                  nb_frames_to_gather_for_computer_agents),
        "random": gather_fixed_frame_nb_input_handler(RandomInputHandler(),
                                                      "random",
                                                      verbose,
                                                      frame_nbs,
                                                      nb_frames_to_gather_for_computer_agents)
    }


def gather_fixed_frame_nb_human(verbose: bool, frame_nbs: List[int]) -> Dict[int, numpy.ndarray]:
    frames = {frame_nb: [] for frame_nb in frame_nbs}
    with open(HUMAN_BC_CUSTOM_A_MODEL_FILE_SPLIT_PATH, "rb") as file:
        split_file_nbs = pickle.load(file)

    test_files = split_file_nbs["test"]
    nb_tests = len(test_files)
    if verbose:
        print("")
    for i in range(nb_tests):
        if verbose:
            print("Reading the frames from human test episode {}/{}...".format(i + 1, nb_tests))
        edl = EpisodeDataLoader(HUMAN_EPISODES_FOLDER, EPISODE_PREFIX, test_files[i])
        for frame_nb in frame_nbs:
            frames[frame_nb].append(cut_off_borders(edl.episode.steps[frame_nb].state.screen))
    return frames


def gather_fixed_frame_nb_input_handler(agent: AInputHandler, agent_name: str, verbose: bool,
                                        frame_nbs: List[int], nb_frames_to_gather: int) -> Dict[int, List[numpy.ndarray]]:
    frames = {frame_nb: [] for frame_nb in frame_nbs}
    max_frame_nb = max(frame_nbs)  # inclusive
    if verbose:
        print("")
    for i in range(nb_frames_to_gather):
        if verbose:
            print("Collecting comparison frames {}/{} for {}...".format(i + 1, nb_frames_to_gather, agent_name))
        env = gym.make("Riverraid-v0")
        initial_observation = env.reset()
        episode = Episode(State(initial_observation, False))

        for frame_nb in range(max_frame_nb + 1):
            action = agent.get_action(episode)
            frame, reward, done, _ = env.step(action)
            episode.add_step(action, reward, State(frame, done))
            if frame_nb in frame_nbs:
                frames[frame_nb].append(cut_off_borders(frame))

    return frames


def cut_off_borders(image: numpy.ndarray) -> numpy.ndarray:
    """Removes black borders and the score/fuel area, so that only the play space remains."""
    return image[2:163, 8:, :]  # manually tuned


def save_fixed_frames_dict(frames_dict: Dict[str, Dict[int, List[numpy.ndarray]]], save_path_without_extension: str):
    folder = os.path.dirname(save_path_without_extension)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(save_path_without_extension + "_frames_dict.xz", "wb") as file:
        compress_pickle.dump(frames_dict, file, compression="lzma")


def analyze_fixed_frames_dict(frames_dict: Dict[str, Dict[int, List[numpy.ndarray]]],
                              save: bool, save_path_base: str):
    for agent_name in frames_dict:
        for frame_nb in frames_dict[agent_name]:
            frames_at_nb = frames_dict[agent_name][frame_nb]

            # Find most central frame in terms of number of distinct pixels (which differ in at least 1 color channel)
            central_frame = None

            min_mean_nb_different_pixels = frames_at_nb[0].shape[0] * frames_at_nb[0].shape[1]
            # (Start with the upper bound of the total number of pixels.)
            for c_frame in frames_at_nb:
                # Try this one as central frame.
                nb_different_pixels_per_frame = [numpy.count_nonzero(numpy.count_nonzero(frame != c_frame, axis=2))
                                                 for frame in frames_at_nb]
                # numpy.count_nonzero(frame != mean_frame, axis=2) is a 2D array of size 210 x 160, where every entry
                # says how in many color channels there was a difference between frame and mean_frame, in a certain
                # pixel location. By taking again numpy.count_nonzero we then get the number of pixels where in at least
                # one color channel there was a difference.
                mean_nb_different_pixels = numpy.mean(nb_different_pixels_per_frame)
                if mean_nb_different_pixels < min_mean_nb_different_pixels:
                    min_mean_nb_different_pixels = mean_nb_different_pixels
                    central_frame = c_frame

            print("For {} the mean number of different pixels with respect to the most central frame "
                  "for frame number {} is {}.".format(agent_name, frame_nb, min_mean_nb_different_pixels))
            if save:
                plt.imsave("{}_{}_CentralFrame_{}.svg".format(save_path_base, agent_name, frame_nb), central_frame)
            else:
                plt.imshow(central_frame)


if __name__ == "__main__":
    """
    RL_BC_agents = load_RL_BC_agents()
    compare_distributions(RL_BC_agents, "RL", True,
                          os.path.join(MODEL_DISTRIBUTION_COMPARISON_FOLDER, "RLvsBCEpisodes_Distr"),
                          True)
    compare_score_and_time_alive(RL_BC_agents, "RL", True, 
                                 os.path.join(MODEL_SCORES_TIME_ALIVE_COMPARISON_FOLDER, "RLvsBCEpisodes_STA"),
                                 True)

    random_BC_agents = load_random_BC_agents()
    compare_distributions(random_BC_agents, "Random", True,
                          os.path.join(MODEL_DISTRIBUTION_COMPARISON_FOLDER, "RandomvsBC_Distr"),
                          True)
    compare_score_and_time_alive(random_BC_agents, "Random", True,
                                 os.path.join(MODEL_SCORES_TIME_ALIVE_COMPARISON_FOLDER, "RandomvsBC_STA"),
                                 True)
    
    compare_distributions_human(True,
                                os.path.join(MODEL_DISTRIBUTION_COMPARISON_FOLDER, "HumanvsBC_Distr"),
                                True)
    compare_score_and_time_alive_human(True,
                                       os.path.join(MODEL_SCORES_TIME_ALIVE_COMPARISON_FOLDER, "HumanvsBCs_STA"),
                                       True)
    """
    compare_fixed_frame_variability(True,
                                    os.path.join(AGENTS_VARIABILITY_COMPARISON_FOLDER, "AgentVariability"),
                                    True, 
                                    [50, 55, 60, 65, 70])
