from collections import deque

import gym
import numpy as np
from PIL import Image

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        self.skip = skip
        self.observation_buffer = deque(maxlen=2)
        super(SkipFrame, self).__init__(env)

    def step(self, action):
        """Step with an action for skip time steps"""
        skip_reward = 0.0
        done = None
        for _ in range(self.skip):
            o, r, done, info = self.env.step(action)
            self.observation_buffer.append(o)
            skip_reward += r
            if done:
                break
        observation_frame = np.max(
            np.stack(self.observation_buffer), axis=0
        )
        return observation_frame, skip_reward, done, info

    def reset(self):
        """Reset the environment and return the next frame"""
        self.observation_buffer.clear()
        o = self.env.reset()
        self.observation_buffer.append(o)
        return o

class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(ProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame.process(obs)

    @staticmethod
    def process(frame, crop=True):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        elif frame.size == 224 * 240 * 3:  # mario resolution
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution." + str(frame.size)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        size = (84, 110 if crop else 84)
        resized_screen = np.array(Image.fromarray(img).resize(size, resample=Image.BILINEAR), dtype=np.uint8)
        x_t = resized_screen[18:102, :] if crop else resized_screen
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ExtraTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps > self._max_episode_steps:
            done = True
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()
