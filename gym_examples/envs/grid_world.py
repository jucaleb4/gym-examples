import gymnasium as gym
from gymnasium import spaces
# import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10, num_obstacles=0, action_eps=0.):
        self.action_eps = action_eps # probability fo doing random action
        self.size = size       # The size of the square grid
        self.window_size = 512 # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        # create obstacles, which upon landing, incurs a large penalty
        rng = np.random.default_rng(0)
        # number of obstacles cannot take up more than 20% of the grid 
        num_obstacles = int(min(np.floor(size*size/5), num_obstacles))
        obstacles_flat = rng.choice(size*size, size=num_obstacles, replace=False)
        obstacles_x = np.mod(obstacles_flat, size)
        obstacles_y = np.floor_divide(obstacles_flat, size)
        self.obstacles_dt = {}
        for (x,y) in zip(obstacles_x, obstacles_y):
            self.obstacles_dt['(%i,%i)' % (x,y)] = True

        all_points = [(x,y) for x in range(size) for y in range(size)]
        self.feasible_points = list(filter(
            lambda c : '(%i,%i)' % (c[0],c[1]) not in self.obstacles_dt, 
            all_points
        ))

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
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            ),
            "obstacles": str(self.obstacles_dt.keys()),
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._agent_location = self.np_random.choice(self.feasible_points, size=1)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            # self._target_location = self.np_random.integers(
            #     0, self.size, size=2, dtype=int
            # )
            self._target_location = self.np_random.choice(self.feasible_points, size=1)

        if isinstance(options, dict) and "s_0" in options:
            self_agent_location = options["s_0"][:2]
            self_target_location = options["s_0"][2:]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        terminated_before_step = np.array_equal(self._agent_location, self._target_location)
        reward = 0 if terminated_before_step else -1  # Binary sparse rewards

        # Randomly perturb action
        if self.np_random.random() <= self.action_eps:
            action = self.np_random.integers(self.action_space.n)

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # but 5% chance we randomly move
        if self.np_random.random() <= 0.05:
            action = self.action_space.sample()

        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # check if we stepped on an obstacle
        agent_loc_str = '(%i,%i)' % (self._agent_location[0,0], self._agent_location[0,1])
        if agent_loc_str in self.obstacles_dt:
            assert not terminated, "Target cannot be same as obstacle"
            reward = -5

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        return 
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pass
            # pygame.display.quit()
            # pygame.quit()
