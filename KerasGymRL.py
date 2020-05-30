
import os

from RLTrainer import SimpleRLTrainer, GreedyRLTrainer
from GlobalSettings import *

if __name__ == "__main__":
    """
    # For SimpleRLTrainer
    trainer = SimpleRLTrainer(load_pi_predict_model=False,
                              load_v_predict_model=False,
                              save_episodes=True,
                              save_gifs=False,
                              save_pi_predict_models=True,
                              save_v_predict_models=True,
                              max_nb_episodes=10000,
                              save_gifs_and_models_frequency=100)
    """
    # For GreedyRLTrainer
    if len(os.listdir(os.path.join(IMPROVING_GREEDY_RL_BASE_FOLDER, RL_MODEL_FOLDER_NAME))) == 2:
        initial_agent_score_estimate = 2100  # From CompareModels, for the fixed RL (= initialisation)
    else:
        initial_agent_score_estimate = None
    trainer = GreedyRLTrainer(initial_agent_score_estimate=initial_agent_score_estimate,
                              save_episodes=True,
                              save_gifs=False,
                              max_nb_episodes=10000,
                              training_block_nb_episodes=25,
                              verbose=1)

    trainer.train()


