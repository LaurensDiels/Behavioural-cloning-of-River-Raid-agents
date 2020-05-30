import numpy
import pygame
from pygame import QUIT

from GlobalSettings import *

from Riverraid import ATARI_FPS, EPISODE_PREFIX
from AtariActions import atari_action_to_name, A_NOOP
from DataLoader import StateActionDataLoader, EpisodeDataLoader

def main_WR():

    ##############
    # Settings:
    EPISODES_FOLDER = RANDOM_EPISODES_FOLDER
    EPISODE_NB = 510
    SPEED_UP_FACTOR = 2
    SKIP_INITIAL_NOOPS = True
    ##############

    # Load in saved data.

    # data_loader = StateActionDataLoader(STATES_FOLDER, ACTION_FOLDER, EPISODE_PREFIX, EPISODE_NB)
    data_loader = EpisodeDataLoader(EPISODES_FOLDER, EPISODE_PREFIX, EPISODE_NB)
    pixels = data_loader.load_states()
    actions = data_loader.load_actions()

    pygame.init()
    fps_clock = pygame.time.Clock()
    replay_fps = SPEED_UP_FACTOR * ATARI_FPS
    pixels = numpy.transpose(pixels, [0, 2, 1, 3])  # pygame uses transposed coordinates compared to numpy
    screen = pygame.display.set_mode(pixels.shape[1:3])
    past_initial_noops = False

    for i in range(pixels.shape[0]):

        if actions[i] != A_NOOP:
            past_initial_noops = True

        if SKIP_INITIAL_NOOPS and not past_initial_noops:
            continue

        print(atari_action_to_name(actions[i]))

        surf_image = pygame.surfarray.make_surface(pixels[i])
        screen.blit(surf_image, (0, 0))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
        fps_clock.tick(replay_fps)


if __name__ == "__main__":
    main_WR()
