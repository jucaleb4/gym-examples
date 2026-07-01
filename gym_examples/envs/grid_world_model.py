import gymnasium as gym
from gymnasium import spaces
# import pygame
import numpy as np

class GridWorldModelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    DIRS = [(1,0), (0,1), (-1,0), (0,-1)]

    def __init__(self, render_mode=None, length=10, n_traps=10, eps=0.05, seed=0):
        self.size = length       # The size of the square grid
        self.window_size = 512 # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, length-1, shape=(2,), dtype=int),
        })

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

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

        self.length = length
        n_states = length*length
        n_actions = 4
        n_traps = min(n_traps, n_states-1)

        rng = np.random.default_rng(seed)
        rnd_pts = rng.choice(length*length, replace=False, size=n_traps+1)
        self.traps = rnd_pts[:-1]
        self.target = target = rnd_pts[-1]
        print("GW: Target at index %d" % self.target)
        print("GW: Traps at ", self.traps)

        P = np.zeros((n_states, n_states, n_actions), dtype=float)
        c = np.zeros((n_states, n_actions), dtype=float)

        def fill_gw_P_at_xy(P, x, y):
            """ 
            Applies standard probability in the 4 cardinal directions provided by @x and @y 

            :param x: x-axis locations of source we want to move from
            :param y: y-axis locations of source we want to move from
            :param length: length of x and y-axis
            :param eps: random probability of moving in another direction
            """
            s = length*y+x
            for a in range(4):
                next_x = np.clip(x + self.DIRS[a][0], 0, length-1)
                next_y = np.clip(y + self.DIRS[a][1], 0, length-1)
                next_s = length*next_y+next_x
                P[next_s, s, a] = (1.-eps)

                # random action
                for b in range(4):
                    if b==a: continue
                    next_x = np.clip(x + self.DIRS[b][0], 0, length-1)
                    next_y = np.clip(y + self.DIRS[b][1], 0, length-1)
                    next_s = length*next_y+next_x
                    P[next_s, s, a] += eps/3 # add to not over-write

        # handle corners
        for i in range(4):
            x = (length-1)*(i%2)
            y = (length-1)*(i//2)
            fill_gw_P_at_xy(P, x, y)

        # vertical edges
        for i in range(2):
            x = (length-1)*i
            y = np.arange(1,length-1)
            fill_gw_P_at_xy(P, x, y)

        # horizontal edges
        for i in range(2):
            y = (length-1)*i
            x = np.arange(1,length-1)
            fill_gw_P_at_xy(P, x, y)

        # inner squares
        x = np.kron(np.ones(length, dtype=int), np.arange(1, length-1))
        y = np.kron(np.arange(1, length-1), np.ones(length, dtype=int))
        fill_gw_P_at_xy(P, x, y)

        # target
        rnd_pts = rng.choice(length*length, replace=False, size=n_traps+1)
        non_target_nor_trap = np.setdiff1d(np.arange(length*length), rnd_pts)

        P[:,self.target,:] = 0
        # go to random non-target non-trap location
        P[non_target_nor_trap,target,:] = 1./len(non_target_nor_trap)

        # apply trap cost
        c[:,:] = 1.
        c[self.traps,:] = 10.
        c[self.target,:] = -10.

        self.r = -c # reward maximization
        self.P = P
        self.agent = None

    def _get_obs(self):
        agent_2d_location = np.array([self.agent//self.length, self.agent%self.length])
        return {"agent": agent_2d_location}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        """ For this ergodic model, do not reset """
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        if self.agent is None:
            self.agent = self.target
            while (self.agent == self.target) or (self.agent in self.traps):
                self.agent = self.np_random.integers(self.length**2)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        reward = self.r[self.agent, action]
        self.agent = self.np_random.choice(self.length*self.length, p=self.P[:,self.agent,action])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        terminated = False
        truncated  = False
        return observation, reward, terminated, truncated, info

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
