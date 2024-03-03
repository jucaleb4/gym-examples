import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleWorldEnv(gym.Env):
    """ Simple environment where the state evolves as

    s_{t+1} = (s_t+a_t+1) mod 2n
    
    reward is s_t + a_t
    """
    metadata = {}

    def __init__(self, size=1, **kwargs):
        self.n = int(size)
        self.state = 0
        self.observation_space = spaces.Discrete(2*self.n)
        self.action_space = spaces.Discrete(self.n)

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):

        reward = self.state + action
        observation = self._get_obs()
        self.state = (self.state + action + 1) % (2*self.n)

        observation = self._get_obs()
        info = self._get_info()
        terminated = truncated = False

        return observation, reward, terminated, truncated, info
