import gymnasium as gym
from gymnasium import spaces
# import pygame
import numpy as np


class LQREnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(low=float('inf'), high=float('inf'), shape=(5,))
        self.action_space = spaces.Box(low=-1e6, high=1e6, shape=(4,))
        self.state = ...

        """ Control of Boeing 747 """
        self.A = np.array([
            [1,-1.1267,-0.6528,-8.0749, 1.5890],
            [0, 0.7741, 0.3176,-0.9772,-2.9690],
            [0, 0.1157, 0.0201,-0.0005,-0.3628],
            [0, 0.0111, 0.0033,-0.0349,-0.0447],
            [0, 0.1388,-0.0862, 0.2935, 0.7579],
        ])
        self.B = np.array([
            [89.1973,-50.1685, 1.1267,-19.3472],
            [ 5.2231,  6.3614, 0.2259, -0.3176],
            [-9.4731,  5.9294,-0.1157,  0.9799],
            [-0.3236,  0.3178,-0.0111, -0.0033],
            [-4.5318,  3.2146,-0.1388,  0.0862],
        ])

        (n,k) = self.B.shape
        self.Q = np.eye(n)
        self.R = np.eye(k)
        # Cov = 0.1 * np.eye(n)
        U = np.array([
            [1, -0.01, 0.5, -0.5, -0.5],
            [0, 1, 0.1, -0.01, -0.01],
            [0, 0, 1, -0.5, -0.5],
            [0, 0, 0, 1, 0.5],
            [0,0,0,0,1]
        ])
        self.Cov = U@U.T

    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.state = self.np_random.normal(
            loc=0, 
            scale=0.1, 
            size=self.observation_space._shape[0]
        )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        x = self.state
        u = action
        reward = -(np.dot(x, self.Q@x) + np.dot(u, self.R@u))

        w = self.np_random.multivariate_normal(mean=np.zeros(len(x)), cov=self.Cov)

        self.state = self.A@x + self.B@u + w

        observation = self._get_obs()
        info = self._get_info()
        terminated = truncated = False

        return observation, reward, terminated, truncated, info

    def close(self):
        pass
