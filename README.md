# Behavioural-cloning-of-River-Raid-agents

All code was written by Laurens Diels in the context of his Master's thesis in Artificial Intelligence (KU Leuven, 2019-2020).

In order for CompareModels.py to run properly, KS2D.py from https://github.com/Gabinou/2DKS (Gabriel Taillon) is also required.


Riverraid.py can be used to let agents (keras or keras-gym models, or human players) play the game and to save episodes of them playing. These can be watched using WatchReplay.py. BehavioralCloner.py is used to behaviorally clone these agents based on saved episodes. Finally the resulting clones can be evaluated using CompareModels.py.

(Although not used in the end, KerasGymRL.py is used to train reinforcement learning agents using keras-gym (Kristian Holsheimer, https://github.com/KristianHolsheimer/keras-gym). However, with the used settings the RL agents started rather quickly to degrade in performance after reaching maximal scores or around 2000 points.)
