import os
import time
from pathlib import Path

import neat
import retro
from Capstone.utils import wrappers as utils

# Constants
from Capstone.NEAT.generate_smb_screenspace import ScreenSpace

runs_per_net = 5


# Use direct memory values as input
def eval_genome_with_memory_inputs(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Create gym env
    env = retro.make(game='SuperMarioBros-Nes')
    buttons = env.buttons

    env = utils.LimitedDiscreteActions(
        utils.ProcessFrame(
            utils.SkipFrame(
                utils.TimeLimit(
                    env, max_episode_steps=4000
                )
            )
        ), buttons
    )

    obs = env.reset()

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
    env.close()
    # The genome's fitness is its worst performance across all runs.
    return min(run_scores)


# Use the screen space as input
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Create gym env
    env = retro.make(game='SuperMarioBros-Nes')
    buttons = env.buttons

    env = utils.LimitedDiscreteActions(
        utils.ProcessFrame(
            utils.SkipFrame(
                utils.TimeLimit(
                    env, max_episode_steps=4000
                )
            )
        ), buttons
    )

    obs = env.reset()

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
    screen_space.print_inputs()
    return screen_space.get_inputs()


def run(use_memory_values=True):
    # Create gym env
    env, obs = utils.create_smb_nes_env()

    done = False
    info = None
    cum_reward = 0

    while True:
        # Test model
        while not done:
            max_value = -2
            max_index = 0

            obs, rew, done, info = env.step(12)
            env.render()
            time.sleep(1 / 60)
            cum_reward = cum_reward + rew
            print(rew)
        print(cum_reward)
        obs = env.reset()
        done = False



def main():
    run()


if __name__ == "__main__":
    main()
