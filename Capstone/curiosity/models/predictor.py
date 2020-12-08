import torch.optim as optim
import torch.nn as nn
import random


class Predictor(nn.Module):
    def __init__(self, input_shape, hidden):
        super(Predictor, self).__init__()

        # Predict next state with this action+observation; p(s|a)
        #self.lstm =
        self.perceptor = nn.Sequential(  # fixme: Maybe need more layers
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, input_shape)
        )

    def forward(self, p_tensor):
        p = self.perceptor(p_tensor)
        return p


class PredictorWrapper():
    def __init__(self, predictor, epochs=2, lr=0.0001, size=8000, load=None):
        self.predictor = predictor
        self.buffer = []
        self.size = size
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.SGD(self.predictor.parameters(), lr=lr)
        if load is not None:
            self.optimizer.load_state_dict(load)
        self.epochs = epochs

    def forward(self, state):
        perception = self.predictor.forward(state)
        return perception

    def append(self, states):
        # Only want state and state prime
        self.buffer.append(states)
        if len(self.buffer) > self.size:
            self.buffer.pop()

    def train(self):
        random.shuffle(self.buffer)
        for _ in range(self.epochs):
            for state, next_state in self.buffer[:-int(len(self.buffer)*0.25)]:
                prediction = self.predictor.forward(state)
                loss = self.loss_function(prediction, next_state)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
