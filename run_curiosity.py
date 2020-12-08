from Capstone.utils import wrappers, Batch, common
from Capstone.curiosity.models import cnn_models, ppo_model
from Capstone.curiosity.models import predictor as p

import torch
from torch.distributions.categorical import Categorical
import numpy as np
import retro
import gc

# INIT default Hyperparameter values
ENV_STEPS = 2450 # Maximum steps allowed in the environment
TRAJECTORY = 32  # Run this number of games before updating
BATCH_SIZE = 8 # Size of Batch taken from trajectory, only for Selective Batch options
MINIBATCH_SIZE = 128 # Size of minibatches from batch
LAMBDA = 0.96 # penalty
EPSILON = 0.2 # limit surrogate loss 2 size
GAMMA = 0.95 # how influential is each step is
EPOCHS = 4
DISCOUNT = 0.5 # how valuable are future states
ENTROPY = 0.001 # encourage new strategy
LEARNING_RATE = 0.0001
device = None # Global device


def get_reward(value, predicted_value):
    """
    Calculate the surprisal value.
    :param value:
        The actual value of the next state. xt+1
    :param predicted_value:
        The predicted value of the next state. (xt|at)
    :return:
        The surprisal reward, prediction error
    :rtype float:
    """

    msq = ((predicted_value - value) ** 2).mean()
    surprisal = -1 * torch.log(msq).squeeze(0)
    return surprisal


def get_features(observation, observer):
    """
    Get feature space from screen space

    :param observation:
        The observed state. xt
    :param observer:
        The network that extracts features
    :return:
        512 logits in a tensor that represent features
    :rtype tensor:
    """
    observation = np.array(observation, copy=False)
    observation = np.expand_dims(observation, 1)
    observation = torch.tensor(observation).to(device).float()
    observation_features = observer.forward(observation)
    return observation_features


def play(env, actor, observer, predictor):
    """
    Play one episode of the environment, getting back the value tuples
    :param env:
        gym environment for the actor to play and the observer to observe
    :param actor:
        PPO network that has an actor, critic, predictors
    :param observer:
        The network that encodes the observation into features for the actor
    :return:
        A list of (observation, action, prediction, observation+1) tuples.
        NOTE! You still have to process the tuples into usable batch data
    :rtype list[tuple]
    """
    total_reward = 0
    observation = env.reset()
    collected_tuples = []
    observation_features = get_features(observation, observer)
    done = False
    iter = 0
    while not done and iter < ENV_STEPS:
        # Get action and predicted features
        actions, value = actor.forward(observation_features)
        prediction = predictor.forward(observation_features.squeeze(0))
        value = value.squeeze(-1).detach()

        action = Categorical(actions).sample()  # Sample as 9 dimensional arrays
        log_prob = Categorical(actions).log_prob(action)
        observation, _, done, info = env.step(action)
        # Save old features for the new tuple in case we're done
        old_features = observation_features
        # Extract New Features
        observation_features = get_features(observation, observer)
        surprisal = get_reward(observation_features, prediction)
        # Save the transition tuple
        predictor.append((old_features, observation_features))
        total_reward += surprisal
        collected_tuples.append((
            old_features,           # state
            action,                 # action
            log_prob,               # log_prob
            surprisal,              # reward
            1 if not done else 0,   # mask
            value                   # V(s)
        ))  # (state, action, log_prob, reward, done_mask, V(s))
        iter += 1

    return collected_tuples


def get_batch_values(raw_batch):
    """
    Calculate the gae and advantages for each transition tuple

    1. mask is 0 if state is terminal, otherwise 1
    2. init GAE to 0, loop backward from last step
    3. delta = r + gamma(V(s`)) * mask - V(s)
    4. GAE = delta + gamma * lambda * gae
    5. return(s,a) = gae + V(s)
    6. reverse back to correct order

    :param raw_batch:
        raw list of tuples from playing an environment
    :return:
        list of tuples with useful values
    :rtype list[tuple]:
    """
    processed_batch = []
    gae = 0
    total_reward = 0
    for r_index in reversed(range(len(raw_batch)-1)):
        state, action, log_prob, reward, mask, value = raw_batch[r_index]
        value_state_prime = raw_batch[r_index+1][-1:][0]
        delta = reward + (GAMMA * value_state_prime * mask) - value_state_prime
        gae = delta + (GAMMA * LAMBDA * gae)
        _return = gae + state
        advantage = value_state_prime - value
        processed_batch.append((state, action, log_prob, _return, advantage))
        total_reward += reward.data.item()
    return reversed(processed_batch), total_reward, (total_reward / len(processed_batch))


def update_policy_gradient(minibatch):
    # Get required variables
    state, action, log_prob, _return, advantage, model_outputs= [],[],[],[],[],[]
    for e in minibatch:
        state.append(e[0])
        action.append(e[1])
        log_prob.append(e[2])
        _return.append(e[3])
        advantage.append(e[4])

    for e in state:
        model_outputs.append(model(e))

    # Get log probability
    new_log_probs = []
    for e in range(len(model_outputs)):
        ta = model_outputs[e][0]
        ta = Categorical(ta).sample(env.action_space.shape)
        new_log_probs.append(Categorical(model_outputs[e][0]).log_prob(ta))

    # Get Loss
    # |  |i
    # || |_
    mapped_prob = [new_log_probs[i] - log_prob[i] for i in range(len(log_prob))]
    ratio = np.exp(mapped_prob)

    surrogate_loss_1 = ratio * advantage
    surrogate_loss_1 = torch.Tensor([x.detach().numpy() for x in surrogate_loss_1])

    ratio = [x.detach().numpy() for x in ratio]
    ratio = torch.Tensor(ratio)
    t_advantage = torch.Tensor([x.detach().numpy() for x in advantage])
    surrogate_loss_2 = torch.clamp(ratio, (1-EPSILON), (1+EPSILON))
    surrogate_loss_2 = [surrogate_loss_2[i] * t_advantage[i] for i in range(len(t_advantage))]
    surrogate_loss_2 = torch.Tensor([x.detach().numpy() for x in surrogate_loss_2])
    actor_loss = - torch.min(surrogate_loss_1, surrogate_loss_2).mean()

    values=[x[1].detach().numpy() for x in model_outputs]
    return_value_difference = [_return[e].detach().numpy() - values[e] for e in range(len(values))]
    critic_loss = np.power(return_value_difference, 2).mean()

    # Get entropy
    actions = [Categorical(x[0]).entropy() for x in model_outputs]
    entropy = sum(actions) / len(actions)

    loss = DISCOUNT * critic_loss + actor_loss - ENTROPY * entropy

    # back propogate the network
    loss.backward(retain_graph=True)
    optimizer.zero_grad()
    optimizer.step()


def test_model(env, actor, observer, predictor):
    observation = env.reset()
    observation_features = get_features(observation, observer)
    total_reward = 0
    done = False
    iter = 0
    while not done:
        # Get action and predicted features
        actions, value = actor.forward(observation_features)
        action = Categorical(actions).sample(env.action_space.shape)  # Sample as 9 dimensional arrays
        observation, _, done, info = env.step(action)
        observation_features = get_features(observation, observer)
        prediction = predictor.forward(observation_features.squeeze(0))
        surprisal = get_reward(observation_features, prediction)
        surprisal = surprisal.squeeze().data.cpu().detach()
        total_reward += surprisal.data.item()
        env.render()
        iter+=1
    return total_reward, (total_reward/iter)


def main(args):
    # Setup for training
    device = torch.device(args.device)
    env = retro.make(game=args.env)
    env = wrappers.create_ppo_smb_nes_env(env)

    if not args.exp:
        BATCH_SIZE = args.batch
    else:
        BATCH_SIZE = int(args.batch * 0.25)

    MINIBATCH_SIZE = args.minibatch
    TRAJECTORY = args.trajectory
    EPOCHS = args.epochs
    ENV_STEPS = args.steps
    batch = Batch.Batch(BATCH_SIZE, MINIBATCH_SIZE)

    # Set up models
    feature_net = cnn_models.Random(env.observation_space.shape, 512).to(device)
    model = ppo_model.ActorCriticPPO(512, env.action_space.n, 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    predictor = p.PredictorWrapper(p.Predictor(512, 512).to(device))
    saver = common.Common('SavedModels/CuriosityModels/', args.name)
    tracker = common.StatTrack('Stats/Curiosity', args.name, args.load_model)
    if args.load_model:
        from pathlib import Path
        model_dict = torch.load(Path('SavedModels/CuriosityModels/' + args.name))
        model.load_state_dict(model_dict['agent'])
        feature_net.load_state_dict(model_dict['cnn'])
        predictor.predictor.load_state_dict(model_dict['per'])
        optimizer.load_state_dict(model_dict['aopt'])
        predictor.optimizer.load_state_dict(model_dict['popt'])
        model.eval()
        predictor.predictor.eval()
    # If new training run, do a random execution and train predictor
    else:
        observation = env.reset()
        observation_features = get_features(observation, feature_net)
        done = False
        iter = 0
        while not done and iter < 2450:
            observation, _, done, info = env.step(env.action_space.sample())
            old_features = observation_features
            observation_features = get_features(observation, feature_net)
            predictor.append((old_features, observation_features))
            iter += 1
        predictor.train()

    # Do main training loop
    iter = 0
    while True:
        # Do TRAJECTORY iterations of the env
        for _ in range(TRAJECTORY):
            iter += 1
            b, t, a = get_batch_values(play(env, model, feature_net, predictor))
            batch.append((b, t))
            tracker.update_iter((t, a))
            print('Iteration Average Reward: ', a)
            print('Environment Iteration: ', iter)
            gc.collect()

        # Update EPOCH times
        for _ in range(EPOCHS):
            for minibatch in batch.iter_minibatch():
                update_policy_gradient(minibatch)

        # Train the predictor on it's own batch
        predictor.train()

        # Get the checkpoint scores
        t, a = test_model(env, model, feature_net, predictor)
        print("Checkpoint Score: ", t)
        print("Checkpoint Average:", a)
        tracker.update_checkpoint((t, a))

        # Save after each batch
        saver.save_torch_agent(model, feature_net, predictor.predictor, optimizer, predictor.optimizer)
        tracker.save_stats()
if __name__ == '__main__':
    # Argument block
    import argparse
    parser = argparse.ArgumentParser(
        description='Main function to run PPO implementation'
    )
    parser.add_argument('--name', type=str, help='Name of files')
    parser.add_argument('--load_model', type=bool, help='Saved model to load', default=False)
    parser.add_argument('--exp', type=bool,
                        help='Run training with experemental settings\n' +
                             'WARNING: If loading model, do not change buffer type',
                        default=False
                        )
    parser.add_argument('--epochs', type=int, help='Number if epochs', default=4)
    parser.add_argument('--batch', type=int, help='Size of the batch', default=32)
    parser.add_argument('--minibatch', type=int, help='Size of minibatch', default=128)
    parser.add_argument('--trajectory', type=int, help='Size of trajectory', default=32)
    parser.add_argument('--steps', type=int, help='Time steps in trajectory', default=2450)
    parser.add_argument('--device', type=str, help='Pytorch device to use during training', default='cpu')
    parser.add_argument('--env', type=str, help='Game environment to run', default='SuperMarioBros-Nes')
    args = parser.parse_args()
    main(args)
