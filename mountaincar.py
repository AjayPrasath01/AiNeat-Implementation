import gym
import neat
import multiprocessing
import pickle
import numpy as np

FastTrain = True
MAX_EPISODE = 500


def eval_genome(genome, config):
    FITNESS = []
    EPISODE = 0
    env = gym.make('MountainCar-v0')
    observation = env.reset()
    net = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)
    while EPISODE < MAX_EPISODE:
        EPISODE += 1
        action = net.activate(observation)
        action = action.index(max(action))
        observation, reward, _, _ = env.step(action)
        if not FastTrain:
            env.render()
        FITNESS.append(observation[0])
    env.close()
    return np.mean(FITNESS)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(gens):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'configurations/MountainCar.txt')

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    if FastTrain:
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count() - 1, eval_genome)
        winner = pop.run(pe.evaluate, gens)
    else:
        winner = pop.run(eval_genomes, gens)
    return winner


def train_without_saving():
    run(1000)


def train_and_save():
    winners = run(1000)
    with open(f'FinishedModels/MountainCar.pkl', 'wb') as file:
        pickle.dump(winners, file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    train_without_saving()
