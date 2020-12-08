"""
Title: Large-Scale Study of Curiosity-Driven Learning
Author: Yuri Burda, Harri Edwards, Deepak Pathak, Amos Storkey, Trevor Darrell and Alexei A. Efros
Date: 2019
Code version: 12/8/2020
Availability: https://github.com/openai/large-scale-curiosity/blob/master/wrappers.py
"""

import itertools
import math
import queue
from collections import deque

import retro
import gym
import gym.spaces
import numpy as np
from PIL import Image

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


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ExtraTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=10000):
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

class SMBMarioFitnessWrapper(gym.Wrapper):
    def __init__(self, env):
        self.max_x_distance = 0
        super(SMBMarioFitnessWrapper, self).__init__(env)

    def step(self, action):
        o, r, done, info = self.env.step(action)
        mario_x = info["player_xpos_high"] * 256 + info["player_xpos_low"]
        if mario_x < self.max_x_distance:
            r = 0
        else:
            self.max_x_distance = mario_x

        return o, r, done, info

    def reset(self):
        self.max_x_distance = 0
        return self.env.reset()

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

class TimeLimitWithRewardThreshold(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None, reward_threshold=0.1, frame_timeout_max=40):
        gym.Wrapper.__init__(self, env)
        self.reward_threshold = reward_threshold
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self.steps_until_done = frame_timeout_max
        self.frame_timeout_max = frame_timeout_max

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        self.steps_until_done = self.steps_until_done - 1

        if reward > 0:
            self.steps_until_done = self.frame_timeout_max

        if self.steps_until_done < 0:
            done = True

        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        self.steps_until_done = self.frame_timeout_max
        return self.env.reset()

class LimitedDiscreteActions(gym.ActionWrapper):
    KNOWN_BUTTONS = {"A", "B"}
    KNOWN_SHOULDERS = {"L", "R"}

    '''
    Reproduces the action space from curiosity paper.
    '''

    def __init__(self, env, all_buttons, whitelist=KNOWN_BUTTONS | KNOWN_SHOULDERS):
        gym.ActionWrapper.__init__(self, env)

        self._num_buttons = len(all_buttons)
        button_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_BUTTONS}
        buttons = [(), *zip(button_keys), *itertools.combinations(button_keys, 2)]
        shoulder_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_SHOULDERS}
        shoulders = [(), *zip(shoulder_keys), *itertools.permutations(shoulder_keys, 2)]
        arrows = [(), (4,), (5,), (6,), (7,)]  # (), up, down, left, right
        acts = []
        acts += arrows
        acts += buttons[1:]
        acts += [a + b for a in arrows[-2:] for b in buttons[1:]]
        self._actions = acts
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons)
        for i in self._actions[a]:
            mask[i] = 1
        return mask

def create_smb_nes_env(max_episode_steps=3000):
    # Create gym env
    env = retro.make('SuperMarioBros-Nes', "Level1-1")
    buttons = env.buttons
    env = TimeLimitWithRewardThreshold(
        SMBMarioFitnessWrapper(
            LimitedDiscreteActions(

                ProcessFrame(
                    SkipFrame(
                        env
                    )
                ), buttons
            )
        ), max_episode_steps=max_episode_steps
    )
    obs = env.reset()
    return env, obs


def create_ppo_smb_nes_env(env):
    env = SkipFrame(env)
    env = ProcessFrame(env)
    env = ExtraTimeLimit(env)
    env = ImageToPyTorch(env)
    env = LimitedDiscreteActions(env, env.buttons)
    return env
=======
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
