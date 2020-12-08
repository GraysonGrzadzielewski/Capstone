import copy
import os
import time
from pathlib import Path

import neat
import retro
import utils

# Constants
from generate_smb_screenspace import ScreenSpace

runs_per_net = 2


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


debug_counter = 0


# Generate an array of memory inputs from a Gym info map
def generate_memory_inputs_with_debug(info):
    screen_space = ScreenSpace(info)
    global debug_counter
    debug_counter = debug_counter + 1
    if debug_counter is 10:
        screen_space.print_inputs()
        debug_counter = 0

    return screen_space.get_inputs()


def run(use_memory_values=True):

    if use_memory_values:
        function_to_run = eval_genome_with_memory_inputs
        config_path = Path('../Configs/config_memory_smbnes.cfg')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    else:
        function_to_run = eval_genome
        config_path = Path('./Configs/config_screen_smbnes.cfg')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

    prefix = "../../../SavedModels/NEATModels/mbnes-generation-"

    checkpointer = neat.checkpoint.Checkpointer(generation_interval=1,
                                                time_interval_seconds=600, filename_prefix=prefix)

    checkpoints = Path.cwd().glob(f"{prefix}*")

    max_checkpoint = -1
    for file_path in checkpoints:
        path_generation = int(file_path.name.split("-")[2])

        if path_generation > max_checkpoint:
            max_checkpoint = path_generation

    if max_checkpoint is not -1:
        pop = checkpointer.restore_checkpoint(f"{prefix}{max_checkpoint}")
    else:
        pop = neat.Population(config)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(6, function_to_run)
    winner = pop.run(pe.evaluate, 1)

    # Show off the winner
    net = neat.nn.RecurrentNetwork.create(winner, config)
    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
    print("Inputs:", inputs)
    print("Outputs:", outputs)

    connections = set()
    for cg in winner.connections.values():
        if cg.enabled:
            connections.add((cg.key[0], cg.key[1]))

    used_nodes = copy.copy(outputs)
    pending = copy.copy(outputs)
    while pending:
        new_pending = set()
        for a, b in connections:
            if b in pending and a not in used_nodes:
                new_pending.add(a)
                used_nodes.add(a)
        pending = new_pending
    print("Used Nodes:", used_nodes)
    for cg in winner.connections.values():
        if cg.enabled:
            _input, output = cg.key
            if _input not in used_nodes or output not in used_nodes:
                continue
            print("Edge: from", _input, "to", output, f"Weight: {cg.weight}")

    # Create gym env
    # Create gym env
    env = retro.make('SuperMarioBros-Nes', "Level1-1")
    buttons = env.buttons

    env = utils.SMBMarioFitnessWrapper(
            utils.LimitedDiscreteActions(
                utils.ProcessFrame(
                    utils.SkipFrame(
                        env
                    )
                ), buttons
            )
        )

    obs = env.reset()

    done = False
    info = None
    cum_reward = 0

    while True:
        if use_memory_values:
            if not info:
                action = [0 for _ in range(0, env.action_space.n)]
            else:
                memory_inputs = generate_memory_inputs_with_debug(info)
                action = net.activate(memory_inputs)
        else:
            action = net.activate(obs.flatten())

        max_value = -2
        max_index = 0

        for index in range(0, len(action)):
            if action[index] > max_value:
                max_value = action[index]
                max_index = index

        obs, rew, done, info = env.step(max_index)
        cum_reward = cum_reward + rew
        time.sleep(1 / 45)
        env.render()
        print(cum_reward)
        if done:
            obs = env.reset()
            cum_reward = 0
            info = None


def main():
    run()


if __name__ == "__main__":
    main()
