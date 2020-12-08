import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


class ActorCriticPPO(nn.Module):
    def __init__(self, input_shape, output_shape, hidden, std=0.0):
        super(ActorCriticPPO, self).__init__()
        # Decide on actions; pi(s|a)
        self.actor = nn.Sequential(  # fixme: Maybe need more layers
            nn.Linear(input_shape, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_shape)
        )
        # Get expected value; V(s)
        self.critic = nn.Sequential( # fixme: Maybe need more layers
            nn.Linear(input_shape, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs):
        actions = self.actor(obs)
        value = self.critic(obs)
        # p_tensor = torch.cat((obs, actions), 1)  # Make new tensor for perceptor
        actions = nn.Softmax(dim=1)(actions)
        #actions = Categorical(actions)
        return actions, value