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
    DIFFERENCE = 3
    SIGMOID_DIFF = 4

    # For Data
    REAL_DATA = 100
    PERIODIC_DATA = 101

class SimpleBatteryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
            self, 
            render_mode=None, 
            battery_capacity=400, 
            transfer_rate=50, 
            start_index=0,
            end_index=-1,
            **kwargs):
        """
        Constructor for simple battery env. We have three modes to describe state/action.

        Current default settings set to: Alamitos Energy Center.

        The state space is represented as

        :param battery_capacity: Battery capacity in MWh
        :param transfer_rate: Battery transfer rate in 
        :param start_index: starting point of data
        :param end_index: ending point of data (-1 means we go until the end of data)
        :param more_data: to include additional data, like DAM demand, solar, and wind


        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        assert transfer_rate <= battery_capacity, f"Transfer rate {transfer_rate} cannot exceed battery capcity {battery_capacity}"
        assert 0 <= start_index, f"Invalid start index {start_index}"
        assert end_index == -1 or start_index < end_index, f"Invalid end index {end_index} > start index {start_index}"

        self.transfer_rate = transfer_rate
        self.battery_capacity = battery_capacity 
        self.battery_storage = 0 
        self.bought_prices = []
        self.ncharges_left = 0
        self.num_consecutive_idle_steps = 0

        self.more_data = bool(kwargs.get("more_data", False))
        self.nhistory = int(kwargs.get("nhistory", 16))
        self.nhistory_hour = int(self.nhistory/4)
        self.avoid_penalty = bool(kwargs.get("avoid_penalty", False))
        self.delay_cost = bool(kwargs.get("delay_cost", False))
        self.daily_cost = float(kwargs.get("daily_cost", 0))
        self.mode = Mode.DEFAULT
        self.data = Mode.REAL_DATA
        for key, val in kwargs.items():
            if key == "mode" and val == "fine_control":
                self.mode = Mode.FINE_CONTROL
            elif key == "mode" and val == "long_charge":
                self.mode = Mode.LONG_CHARGE
            elif key == "mode" and val == "difference":
                self.mode = Mode.DIFFERENCE
            elif key == "mode" and val == "sigmoid":
                self.mode = Mode.SIGMOID_DIFF
            elif key == "data" and val == "periodic":
                self.data = Mode.PERIODIC_DATA

        self.load_data()
        self.penalty_rate = -np.max(self.lmp_arr)/20
        self.setup_observation_space(start_index, end_index)
        self.setup_action_space()

        self.window_size = 512  # The size of the PyGame window
        self.window = None
        self.clock = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def load_data(self):
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

    def setup_observation_space(self, start_index, end_index):
        """
            - DEFAULT: Normal on/off charging at 15 min interval
            - FINE_CONTROL: More fine-grained charging
            - LONG_CHARGE: Allowing chrage for 1-3 full days (enlarge state space to keep track of remaining charging days)

        The base state space consists of:
            1. battery state of charge (SOC) (1)
            2. current and past LMPs in chronologically decreasing order (nhistory)
        If we want to include more information, we append the state space by
            3. demands (floor(nhistory/4) - divide by 4 is demand is DA (1hr) while LMP is RT (15min))
            4. solar (floor(nhistory/4))
            5. wind (floor(nhistory/4))

        If we use mode `difference` or `sigmoid_diff`, we show the LMP and
        forecast differences between consecutive values. The former shows the
        raw difference while the latter shows the difference with a sigmoid,
        i.e., 1/(1+exp(-x)). We also normalize the SOC to be between [0,1],
        and it is the fraction that the battery is full.
        """
        self.start_index = min(start_index, len(self.lmp_arr)-1)
        self.end_index = len(self.lmp_arr) if end_index == -1 else min(end_index, len(self.lmp_arr))
        self.time_step = self.start_index

        lows = np.append(0, [np.min(self.lmp_arr)]*self.nhistory)
        highs = np.append(self.battery_capacity, [np.max(self.lmp_arr)]*self.nhistory)
        if self.more_data:
            lows = np.append(lows, 
                [np.min(self.demand_arr)] * self.nhistory_hour
                + [np.min(self.solar_arr)] * self.nhistory_hour
                + [np.min(self.wind_arr)] * self.nhistory_hour
            )
            highs = np.append(highs, 
                [np.max(self.demand_arr)] * self.nhistory_hour
                + [np.max(self.solar_arr)] * self.nhistory_hour
                + [np.max(self.wind_arr)] * self.nhistory_hour
            )
        if self.mode == Mode.LONG_CHARGE:
            lows = np.append(lows, 0)
            highs = np.append(highs, 2)
        elif self.mode == Mode.DIFFERENCE:
            highs = np.append(0, [np.ptp(self.lmp_arr)]*(self.nhistory-1))
            if self.more_data:
                highs = np.append(highs,
                    [np.ptp(self.demand_arr)]*(self.nhistory_hour-1)
                    + [np.ptp(self.solar_arr)]*(self.nhistory_hour-1)
                    + [np.ptp(self.wind_arr)]*(self.nhistory_hour-1)
                )
            lows = -highs
            highs[0] = self.battery_capacity
        elif self.mode == Mode.SIGMOID_DIFF:
            highs = np.ones(self.nhistory)
            if self.more_data:
                highs = np.append(highs, np.ones(3*(self.nhistory_hour-1)))
            lows = 0 * highs

        self.lows = lows
        self.highs = highs
        self.observation_space = spaces.Box(low=lows, high=highs, dtype=float)

    def setup_action_space(self):
        if self.mode == Mode.FINE_CONTROL:
            self.action_space = spaces.Discrete(11)
            self.action_to_direction = {
                0: -self.transfer_rate,
                1: -self.transfer_rate * 0.8,
                2: -self.transfer_rate * 0.6,
                3: -self.transfer_rate * 0.4,
                4: -self.transfer_rate * 0.2,
                5: 0,
                6: self.transfer_rate * 0.2,
                7: self.transfer_rate * 0.4,
                8: self.transfer_rate * 0.6,
                9: self.transfer_rate * 0.8,
                10: self.transfer_rate 
            }
        elif self.mode == Mode.LONG_CHARGE:
            self.action_space = spaces.Discrete(5)
            self.action_to_direction = {
                0: -transfer_rate,
                1: 0,
                2: self.transfer_rate,
                3: self.transfer_rate, # +1 day of charging
                4: self.transfer_rate, # +2 days of charging
            }
        else:
            self.action_space = spaces.Discrete(3)
            self.action_to_direction = {
                0: -self.transfer_rate, # discharge (make money)
                1: 0,              # do nothing
                2: self.transfer_rate   # charge (lose money)
            }

    def _get_obs(self):
        rt_idx = self.time_step 
        da_idx = int(self.time_step/4) 

        obs = np.array([self.battery_storage])
        recent_lmps = self.lmp_arr.take(
            np.arange(rt_idx-self.nhistory+1,rt_idx+1), 
            mode="wrap")[::-1]
        obs = np.append(obs, recent_lmps)

        if self.more_data:
            recent_demand = self.demand_arr.take(
                np.arange(da_idx-self.nhistory_hour+1,da_idx+1), 
                mode="wrap")[::-1]
            recent_solar = self.solar_arr.take(
                np.arange(da_idx-self.nhistory_hour+1,da_idx+1), 
                mode="wrap")[::-1]
            recent_wind = self.wind_arr.take(
                np.arange(da_idx-self.nhistory_hour+1,da_idx+1), 
                mode="wrap")[::-1]
            obs = np.append(obs, recent_demand) 
            obs = np.append(obs, recent_solar) 
            obs = np.append(obs, recent_wind)

        if self.mode == Mode.LONG_CHARGE:
            obs = np.append(obs, self.ncharges_left)
        elif self.mode in [Mode.DIFFERENCE, Mode.SIGMOID_DIFF]:
            obs = np.array([self.battery_storage])
            obs = np.append(obs, np.ediff1d(recent_lmps))
            if self.more_data:
                obs = np.append(obs, np.ediff1d(recent_demand))
                obs = np.append(obs, np.ediff1d(recent_solar))
                obs = np.append(obs, np.ediff1d(recent_wind))
            if self.mode == Mode.SIGMOID_DIFF:
                obs[0] /= self.battery_capacity
                obs[1:] = np.divide(1., 1. + np.exp(-obs[1:]))

        return obs

    def _get_info(self):
        return {"time_step": self.time_step}

    def step(self, action):
        # adjust action based on battery constraints 
        if self.mode == Mode.LONG_CHARGE and self.ncharges_left > 0:
            action = 2 
        max_charge = self.battery_storage >= self.battery_capacity 
        min_charge = self.battery_storage <= 0
        if (action == 2 and max_charge) or (action == 0 and min_charge):
            action = 1
        if self.mode == Mode.LONG_CHARGE:
            if action == 3: # TODO: Magic number
                self.ncharges_left = 1
            elif action == 4: # TODO: Magic number
                self.ncharges_left = 2
            else:
                self.ncharges_left = max(0, self.ncharges_left-1)

        # Update system variables
        transfer = self.action_to_direction[action]
        past_battery_storage = self.battery_storage
            
        self.battery_storage = np.clip( past_battery_storage + transfer, 0, self.battery_capacity)
        battery_change = self.battery_storage - past_battery_storage
        current_lmp = self.lmp_arr[self.time_step]
        if self.time_step < self.end_index-1: 
            self.time_step += 1
        else:
            self.time_step = self.start_index

        reward = current_lmp * (-battery_change)
        if self.delay_cost:
            reward = 0
            if action == 2: 
                self.bought_prices.append(current_lmp)
            elif action == 0:
                assert len(self.bought_prices) > 0, "Internal code error: cannot sell if bought_prices is empty for delay_cost setting"
                oldest_bought_lmp = self.bought_prices.pop(0)
                reward = (current_lmp - oldest_bought_lmp) * self.transfer_rate
        reward -= self.daily_cost

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

        self.battery_storage = 0
        self.bought_prices = []
        if options is not None and options.get("rand_start", False):
            self.time_step = self.np_random.integers(self.start_index, self.end_index, dtype=int)
        elif options is not None:
            self.time_step = options.get("start", 0) % (self.end_index - self.start_index) + self.start_index
        else:
            self.time_step = self.start_index

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render_frame()

        return observation, info

    def close(self):
        if self.window is not None:
            pass
            # pygame.display.quit()
            # pygame.quit()

