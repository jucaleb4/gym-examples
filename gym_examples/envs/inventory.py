import gymnasium as gym
from gymnasium import spaces
# import pygame
import numpy as np


class InventoryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, n_actions=10):
        self.window_size = 512 # The size of the PyGame window

        # inventory params
        self.h = 0.2 # holding cost
        self.b = 2.8 # backlog cost
        self.c = np.cos(np.pi/3) + 1.5 # ordering cost
        self.d = 10.0 # demand mean
        self.phi = 1.6 # demand std
        self.q_lim = 40 # quantity limit

        # continuous (inventory after replenishment and before demand)
        self.observation_space = spaces.Box(low=-self.q_lim, high=self.q_lim, shape=(1,), dtype=np.float32)
        self._inventory = 0

        # discrete (replenishment)
        self.action_space = spaces.Discrete(n_actions)
        self.replenishment_arr = np.linspace(0, 2*self.q_lim, num=n_actions, endpoint=True)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return np.array([self._inventory], dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose empty stockpile
        self._inventory = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # y^t = x^{t-1} - D^t (inventory after demand)
        D_t = max(0, self.np_random.normal(loc=self.d, scale=self.phi))
        y_t = self._inventory - D_t

        # x^t = y^t + o^t (clip between [-q_lim, q_lim])
        o_t = self.replenishment_arr[action]
        self._inventory = o_t + y_t
        self._inventory = max(-self.q_lim, min(self.q_lim, self._inventory))
        actual_o_t = self._inventory - y_t

        cost = self.c * actual_o_t + self.h * max(0, y_t) + self.b * max(0, -y_t)
        reward = -cost

        observation = self._get_obs()
        info = self._get_info()
        terminated = False

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        return 

    def close(self):
        return
