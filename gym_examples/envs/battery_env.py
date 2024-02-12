import numpy as np
# import pygame

import gymnasium as gym
from gymnasium import spaces
from gym_examples.envs.parse import get_caiso_data

from enum import Enum

class Mode(Enum):
    # For action types
    DEFAULT = 0
    FINE_CONTROL = 1
    LONG_CHARGE = 2
    PENALIZE = 3
    DELAY = 4
    QLEARN = 5
    PENALIZE_FULL = 6
    PENALIZE_WAIT = 7

    # For Data
    REAL_DATA = 100
    PERIODIC_DATA = 101

class SimpleBatteryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, battery_capacity=400, transfer_rate=50, **kwargs):
        """
        Constructor for simple battery env. We have three modes to describe state/action.

            - DEFAULT: Normal on/off charging at 15 min interval
            - FINE_CONTROL: More fine-grained charging
            - LONG_CHARGE: Allowing chrage for 1-3 full days (enlarge state space to keep track of remaining charging days)
            - PENALIZE: Penalize (enlarge state space to keep track of last day we charged) with maximum of 20 days worth of no charging
            - DELAY: Delay penalty (enlarge by keeping number of assets and average cost)

        :params battery_capacity: Battery capacity in MWh
        :params transfer_rate: Battery transfer rate in 

        Current default settings set to: Alamitos Energy Center

        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        # First get mode we want and other parameters
        self.mode = Mode.DEFAULT
        self.data = Mode.REAL_DATA
        self.nhistory = 10
        for key, val in kwargs.items():
            if key == "mode" and val == "fine_control":
                self.mode = Mode.FINE_CONTROL
            elif key == "mode" and val == "long_charge":
                self.mode = Mode.LONG_CHARGE
            elif key == "mode" and val == "penalize":
                self.mode = Mode.PENALIZE
            elif key == "mode" and val == "delay":
                self.mode = Mode.DELAY
            elif key == "mode" and val == "qlearn":
                self.mode = Mode.QLEARN
            elif key == "mode" and val == "penalize_full":
                self.mode = Mode.PENALIZE_FULL
            elif key == "mode" and val == "penalize_wait":
                self.mode = Mode.PENALIZE_WAIT

            if key == "data" and val == "periodic":
                print("Switching to periodic dataset")
                self.data = Mode.PERIODIC_DATA

            elif key == "nhistory":
                self.nhistory = int(val)

        self.window = None
        self.clock = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # system variables
        self.battery_storage = 0 # battery_capacity / 2
        self.num_buys = 0
        self.max_buys = min(12, int(np.ceil(battery_capacity/transfer_rate)))
        self.steps_since_last_sold = 0
        self.bought_prices = np.zeros(self.max_buys, dtype=float)
        self.bought_prices_pt = 0
        self.battery_capacity = battery_capacity 
        self.transfer_rate = transfer_rate
        self.window_size = 512  # The size of the PyGame window

        # get data
        self.time_step = 0
        if self.data == Mode.REAL_DATA:
            lmp_arr, demand_arr, solar_arr, wind_arr = get_caiso_data()
            self.lmp_arr = lmp_arr
            self.demand_arr = demand_arr * 1/4 # convert MW -> MWh (15 min interval)
            self.solar_arr = solar_arr
            self.wind_arr = wind_arr
        elif self.data == Mode.PERIODIC_DATA:
            [lo, hi] = [-225, 725]
            ndata = 90 * 4 * 24
            self.lmp_arr = (hi-lo)/2 * (np.sin(np.arange(ndata) * 2 * np.pi/(4*24)) + 1) + (hi+lo)/2

        # --- Observations are tuple of 5 values: battery, LMP, demand, solar, wind, average cost ---
        # Observations are tuple of 5 values: battery, current LMP, historic LMP
        lows = np.append(0, np.min(self.lmp_arr)*np.ones(self.nhistory))
        highs = np.append(battery_capacity, np.max(self.lmp_arr)*np.ones(self.nhistory))
        # self.avg_price = 0

        if self.mode == Mode.LONG_CHARGE:
            lows = np.append(lows, 0)
            highs = np.append(highs, 2)
            self.ncharges_left = 0
        elif self.mode == Mode.PENALIZE:
            # TODO: Added average cost of battery
            # lows = np.append(lows, [0, np.min(self.lmp_arr)])
            # highs = np.append(highs, [20, np.max(self.lmp_arr)])
            lows = np.append(lows, 0)
            highs = np.append(highs, 20)
            self.penalty_rate = -np.max(self.lmp_arr)/20
            self.num_consecutive_idle_steps = 0
        elif self.mode == Mode.DELAY or self.mode == Mode.PENALIZE_FULL:
            # append number of buys and last LMP
            lows = np.append(lows, [0, min(0, np.min(self.lmp_arr))])
            highs = np.append(highs, [self.max_buys, np.max(self.lmp_arr)])
        elif self.mode == Mode.QLEARN:
            lows = np.zeros(self.nhistory-1)
            highs = np.ones(self.nhistory-1)
        elif self.mode == Mode.PENALIZE_WAIT:
            # same Mode.PENALIZE_FULL with integer for distance to last sold if
            # battery not empty (penalize up to 1 day, 96 15m intervals)
            lows = np.append(lows, [0, min(0, np.min(self.lmp_arr)), 0])
            highs = np.append(highs, [self.max_buys, np.max(self.lmp_arr), 96])

        self.observation_space = spaces.Box(low=lows, high=highs, dtype=float)

        if self.mode == Mode.FINE_CONTROL:
            self.action_space = spaces.Discrete(11)
            self.action_to_direction = {
                0: -transfer_rate,
                1: -transfer_rate * 0.8,
                2: -transfer_rate * 0.6,
                3: -transfer_rate * 0.4,
                4: -transfer_rate * 0.2,
                5: 0,
                6: transfer_rate * 0.2,
                7: transfer_rate * 0.4,
                8: transfer_rate * 0.6,
                9: transfer_rate * 0.8,
                10: transfer_rate 
            }
        elif self.mode == Mode.LONG_CHARGE:
            self.action_space = spaces.Discrete(5)
            self.action_to_direction = {
                0: -transfer_rate,
                1: 0,
                2: transfer_rate,
                3: transfer_rate, # +1 day of charging
                4: transfer_rate, # +2 days of charging
            }
        else:
            self.action_space = spaces.Discrete(3)
            self.action_to_direction = {
                0: -transfer_rate, # discharge (make money)
                1: 0,              # do nothing
                2: transfer_rate   # charge (lose money)
            }

    # helper functions
    def _get_obs(self):
        rt_idx = self.time_step # real time indexing
        da_idx = self.time_step // 4 # day ahead (hourly) indexing

        obs = np.array([
            self.battery_storage, 
            # self.lmp_arr[rt_idx], 
            # self.demand_arr[da_idx],
            # self.solar_arr[da_idx],
            # self.wind_arr[da_idx],
        ])

        # past `nhistory`-1 (i.e., not including present)
        lmp_last = self.lmp_arr.take(np.arange(rt_idx-self.nhistory+1,rt_idx+1), mode="wrap")
        # reverse from most recent to oldest date
        lmp_last = lmp_last[::-1]
        obs = np.append(obs, lmp_last)

        # obs = np.append(obs, self.avg_price)

        if self.mode == Mode.LONG_CHARGE:
            obs = np.append(obs, self.ncharges_left)
        if self.mode == Mode.PENALIZE:
            obs = np.append(obs, self.num_consecutive_idle_steps)
        if self.mode == Mode.DELAY or self.mode == Mode.PENALIZE_FULL:
            obs = np.append(obs, [self.num_buys, self.bought_prices[0]])
        if self.mode == Mode.QLEARN:
            lmp_diff = np.diff(lmp_last)
            lmp_diff_sigmoid = np.divide(1., 1. + np.exp(-lmp_diff))
            obs = lmp_diff_sigmoid
        if self.mode == Mode.PENALIZE_WAIT:
            obs = np.append(obs, [self.num_buys, self.bought_prices[0]])

        return obs

    def _get_info(self):
        return {"time_step": self.time_step}

    def step(self, action):
        reward = 0

        # Adjust action based on constraints of system
        if self.mode == Mode.LONG_CHARGE and self.ncharges_left > 0:
            action = 2 # TODO: Magic number

        max_charge = self.battery_storage >= self.battery_capacity
        min_charge = self.battery_storage <= 0
        if (action == 2 and max_charge) or (action == 0 and min_charge):
            # penalize if we tried to buy
            if self.mode == Mode.PENALIZE_FULL and action == 2:
                reward = -np.max(self.lmp_arr) * 0.1

            # do nothing if we cannot purchase more
            action = 1

        # Update system variables
        transfer = self.action_to_direction[action]
        past_battery_storage = self.battery_storage
            
        self.battery_storage = np.clip( past_battery_storage + transfer, 0, self.battery_capacity)
        battery_change = self.battery_storage - past_battery_storage
        current_lmp = self.lmp_arr[self.time_step]

        if battery_change < 0: 
            reward = current_lmp * battery_change
        elif battery_change > 0: 
            reward = current_lmp * (-battery_change)

        if self.mode == Mode.PENALIZE: 
            if battery_change == 0:
                self.num_consecutive_idle_steps += 1

                # TODO: Magic number: start penalizing after 3 consecutive days
                reward = max(self.num_consecutive_idle_steps-2, 0) * self.penalty_rate
            else:
                self.num_consecutive_idle_steps = 0 

        # Update state and variables
        self.time_step = (self.time_step + 1) % len(self.lmp_arr)
        if self.mode == Mode.LONG_CHARGE:
            if action == 3: # TODO: Magic number
                self.ncharges_left = 1
            elif action == 4: # TODO: Magic number
                self.ncharges_left = 2
            else:
                self.ncharges_left = max(0, self.ncharges_left-1)
        if self.mode == Mode.DELAY or self.mode == Mode.QLEARN or self.mode == Mode.PENALIZE_FULL or self.mode == Mode.PENALIZE_WAIT:
            # purchase
            if action == 2: 
                self.bought_prices[self.bought_prices_pt] = current_lmp
                self.bought_prices_pt += 1
                reward = 0
                self.num_buys += 1
            # sell
            elif action == 0:
                # treat self.bought_prices as queue
                oldest_lmp = self.bought_prices[0]
                self.bought_prices[0:self.bought_prices_pt-1] = self.bought_prices[1:self.bought_prices_pt]
                self.bought_prices_pt -= 1
                reward = (current_lmp - oldest_lmp) * abs(battery_change)
                self.num_buys -= 1
                self.steps_since_last_sold = 0
        if self.mode == Mode.PENALIZE_WAIT:
            if action == 1 and self.battery_storage > 0:
                self.steps_since_last_sold = min(1+self.steps_since_last_soldn, 96)
                reward = -np.max(self.lmp_arr) * 0.1 * self.steps_since_last_sold/96

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render_frame()

        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's battery level and starting time randomly
        # self.battery_storage = self.np_random.uniform(0, self.battery_capacity)
        self.battery_storage = 0
        if self.mode == Mode.QLEARN or self.mode == Mode.DELAY:
            self.num_buys = 0
            self.bought_prices[:] = 0
            self.bought_prices_pt = 0

        if options is not None and options.get("rand_start", False):
            self.time_step = self.np_random.integers(0, len(self.lmp_arr), dtype=int)
        elif options is not None:
            self.time_step = options.get("start", 0) % len(self.lmp_arr)
        else:
            self.time_step = 0
        self.niters_since_last_charge = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render_frame()

        if options is not None and options.get("s_0", None) is not None:
            self.time_step = options["s_0"]
            assert 0 <= self.time_step < len(self.lmp_arr), "Initial time step must be in [0, {len(self.lmp_arr)-1}]"

        return observation, info

    def close(self):
        if self.window is not None:
            pass
            # pygame.display.quit()
            # pygame.quit()

