from Capstone.utils import wrappers, Batch, common
from Capstone.curiosity.models import cnn_models, ppo_model
from Capstone.curiosity.models import predictor as p

import torch
from torch.distributions.categorical import Categorical
import numpy as np
import retro

device = None #init global device

def get_features(observation, observer):
    """

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
    surprisal = torch.log(msq).squeeze(0)
    return surprisal

def main(name):
    suprisal_loss = torch.nn.MSELoss()  # Don't make a new object every time
    env = retro.make(game="SuperMarioBros-Nes")
    env = wrappers.create_ppo_smb_nes_env(env)
    device = torch.device("cpu")
    # Set up models
    feature_net = cnn_models.Random(env.observation_space.shape, 512).to(device)
    model = ppo_model.ActorCriticPPO(512, env.action_space.n, 128).to(device)

    # Set up models
    feature_net = cnn_models.Random(env.observation_space.shape, 512).to(device)
    model = ppo_model.ActorCriticPPO(512, env.action_space.n, 128).to(device)
    predictor = p.PredictorWrapper(p.Predictor(512, 512).to(device))
    saver = common.Common('SavedModels/CuriosityModels/', name)
    tracker = common.StatTrack('Stats/Curiosity', name, True)

    from pathlib import Path
    model_dict = torch.load(Path('SavedModels/CuriosityModels/' + name))
    model.load_state_dict(model_dict['agent'])
    feature_net.load_state_dict(model_dict['cnn'])
    predictor.predictor.load_state_dict(model_dict['per'])
    predictor.optimizer.load_state_dict(model_dict['popt'])
    model.eval()
    predictor.predictor.eval()


    while True:
        observation = env.reset()
        observation_features = get_features(observation, feature_net)
        done = False
        while not done:
            actions, value = model.forward(observation_features)
            action = Categorical(actions).sample(env.action_space.shape)
            observation, _, done, info = env.step(action)
            observation_features = get_features(observation, feature_net)
            prediction = predictor.forward(observation_features.squeeze(0))
            surprisal = get_reward(observation_features, prediction)
            surprisal = surprisal.squeeze().data.cpu().detach()
            print(surprisal)
            env.render()
