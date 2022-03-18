import gym
import neat
import multiprocessing
import pickle
import numpy as np
import visuvalize
from time import sleep
from graphviz import Source

import platform
plt = platform.system()

config_path = ""
finished_model_path = ""

if plt == "Windows":
    print("Your system is Windows Code Must be modified for windows")
elif plt == "Linux" or plt == "Darwin":
    config_path = 'configurations/MountainCar.txt'
    finished_model_path = 'FinishedModels/MountainCar.pkl'
else:
    print("Unidentified system")

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

with open(finished_model_path, "rb") as input_file:
    genome = pickle.load(input_file)


def run():
    env = gym.make('MountainCar-v0')
    done = False
    observation = env.reset()
    net = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)
    while not done:
        env.render()
        action = net.activate(observation)
        observation, reward, _, _ = env.step(action.index(max(action)))

    dot = visuvalize.draw_net(config, genome, False, show_disabled=False)
    env.close()
    s = Source(dot, filename='MountainCarNet', format='png')
    s.view()

