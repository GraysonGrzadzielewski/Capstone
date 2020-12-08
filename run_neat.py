import os
from pathlib import Path

import neat
import retro
from Capstone.utils import wrappers as utils
from Capstone.NEAT.MarioStatisticsReporter import MarioStatisticsReporter
from Capstone.NEAT.generate_smb_screenspace import ScreenSpace

# Constants
runs_per_net = 4


# Use direct memory values as input
def eval_genome_with_memory_inputs(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    env, obs = utils.create_smb_nes_env()

    # do runs_per_net runs and get the minimum fitness as the result for this generation
    run_scores = []

    for runs in range(runs_per_net):
        done = False
        info = None
        cum_reward = 0
        # Test model
        while not done:
            if not info:
                action = [0 for _ in range(0, env.action_space.n)]
            else:
                memory_inputs = generate_memory_inputs(info)
                action = net.activate(memory_inputs)

            max_value = -2
            max_index = 0

            for index in range(0, len(action)):
                if action[index] > max_value:
                    max_value = action[index]
                    max_index = index

            obs, rew, done, info = env.step(max_index)
            cum_reward = cum_reward + rew

        env.reset()
        run_scores.append(cum_reward)
    env.close()
    # The genome's fitness is its worst performance across all runs.
    return min(run_scores)


# Use the screen space as input
def eval_genome(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    # Create gym env
    env, obs = utils.create_smb_nes_env()

    # do runs_per_net runs and get the minimum fitness as the result for this generation
    run_scores = []

    for runs in range(runs_per_net):
        done = False
        cum_reward = 0
        # Test model
        while not done:
            action = net.activate(obs.flatten())

            max_value = -2
            max_index = 0

            for index in range(0, len(action)):
                if action[index] > max_value:
                    max_value = action[index]
                    max_index = index

            obs, rew, done, info = env.step(max_index)
            cum_reward = cum_reward + rew

        env.reset()
        run_scores.append(cum_reward)
    env.close()
    # The genome's fitness is its worst performance across all runs.
    return min(run_scores)


# Generate an array of memory inputs from a Gym info map
def generate_memory_inputs(info):
    screen_space = ScreenSpace(info)
    return screen_space.get_inputs()


def run(use_memory_values=True):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)

    if use_memory_values:
        function_to_run = eval_genome_with_memory_inputs
        config_path = Path('./Capstone/NEAT/Configs/config_memory_smbnes.cfg')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    else:
        function_to_run = eval_genome
        config_path = Path('./Capstone/NEAT/Configs/config_screen_smbnes.cfg')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

    prefix = "./SavedModels/NEATModels/smbnes-generation-"

    checkpointer = neat.checkpoint.Checkpointer(generation_interval=1,
                                                time_interval_seconds=600,
                                                filename_prefix=prefix)

    checkpoints = Path.cwd().glob(f"{prefix}*")

    max_checkpoint = -1
    for file_path in checkpoints:
        path_generation = int(file_path.name.split("-")[2])

        if path_generation > max_checkpoint:
            max_checkpoint = path_generation

    if max_checkpoint != -1:
        pop = checkpointer.restore_checkpoint(f"{prefix}{max_checkpoint}")

    else:
        pop = neat.Population(config)

    stats = MarioStatisticsReporter(out_dir='./Stats/NEAT/')
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(checkpointer)

    cores = int(os.cpu_count() / 2) - 1
    pe = neat.ParallelEvaluator(cores, function_to_run)

    winner = pop.run(pe.evaluate, 300)
    stats.save()


def main():
    run()


if __name__ == "__main__":
    main()
