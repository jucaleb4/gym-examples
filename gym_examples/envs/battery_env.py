import warnings

from collections import OrderedDict
from enum import Enum

import numpy as np
import numpy.linalg as la
# import pygame

import gymnasium as gym
from gymnasium import spaces
from gym_examples.envs.parse import get_caiso_data

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
            battery_power=100, 
            start_index=0,
            end_index=-1,
            **kwargs):
        """
        Constructor for simple battery env. We have three modes to describe state/action.

        Current default settings set to: Alamitos Energy Center.

        The state space is represented as

        :param battery_capacity: Battery capacity (MWh)
        :param battery_power: Battery power or transfer rate (MW)
        :param start_index: starting point of data
        :param end_index: ending point of data (-1 means we go until the end of data)
        :param more_data: to include additional data, like DAM demand, solar, and wind


        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        assert 0 <= start_index, f"Invalid start index {start_index}"
        assert end_index == -1 or start_index < end_index, f"Invalid end index {end_index} > start index {start_index}"

        # "state" variables 
        self.rt_scale = 0.25 # since we are operating every 15min
        self.battery_power = battery_power
        self.battery_capacity = battery_capacity 
        self.battery_storage = 0.
        self.avg_bought_lmp = 0. # unit: $/MWh
        self.n_charges_left = 0   # for multi-period buying
        self.num_consecutive_idle_steps = 0 

        # below are settings for the environment
        self.more_data = bool(kwargs.get("more_data", False))
        self.solar_coloc = bool(kwargs.get("solar_coloc", False))
        self.nhistory = int(kwargs.get("nhistory", 16))
        self.nhistory_hour = max(1, int(self.nhistory/4))
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
            lmp_arr, demand_arr, solar_arr, wind_arr, actual_solar_arr = get_caiso_data()
            self.lmp_arr = lmp_arr
            self.demand_arr = demand_arr 
            self.solar_arr = solar_arr
            self.wind_arr = wind_arr
            self.actual_solar_arr = wind_arr

            # we scale down the solar power so that it matches 10% of battery charging
            scale = 0.1 * min(1, self.battery_power/la.norm(self.actual_solar_arr, ord=np.inf))
            self.actual_solar_arr = scale * self.actual_solar_arr
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
        # this will be used to construct the Dict observation space
        dt = OrderedDict()

        self.start_index = min(start_index, len(self.lmp_arr)-1)
        self.end_index = len(self.lmp_arr) if end_index == -1 else min(end_index, len(self.lmp_arr))
        self.time_step = self.start_index

        lows = np.append(0, [np.min(self.lmp_arr)]*self.nhistory)
        highs = np.append(self.battery_capacity, [np.max(self.lmp_arr)]*self.nhistory)
        dt["battery_soc"] = spaces.Box(low=0, high=self.battery_capacity)
        dt["lmps"] = spaces.Box(
            low=np.min(self.lmp_arr), 
            high=np.max(self.lmp_arr), 
            shape=(self.nhistory,),
        )
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
            dt["demand_fcsts"] = spaces.Box(low=np.min(self.demand_arr), high=np.max(self.demand_arr), shape=(self.nhistory,))
            dt["solar_fcsts"] = spaces.Box(low=np.min(self.solar_arr), high=np.max(self.solar_arr), shape=(self.nhistory,))
            dt["wind_fcsts"] = spaces.Box(low=np.min(self.wind_arr), high=np.max(self.wind_arr), shape=(self.nhistory,))
        if self.mode == Mode.LONG_CHARGE:
            lows = np.append(lows, 0)
            highs = np.append(highs, 2)
            dt["remaining_charges"] = spaces.Box(low=0, high=2, shape=())
        elif self.mode == Mode.DIFFERENCE:
            highs = np.append(0, [np.ptp(self.lmp_arr)]*(self.nhistory-1))
            lows = -highs
            highs[0] = self.battery_capacity
            dt = OrderedDict([
                ("battery_soc", spaces.Box(low=0, high=self.battery_capacity)),
                ("lmp_diffs", spaces.Box(
                    low=-np.ptp(self.lmp_arr), 
                    high=np.ptp(self.lmp_arr), 
                    shape=(self.nhistory-1,)
                )),
            ])

            if self.more_data:
                highs = np.append(highs,
                    [np.ptp(self.demand_arr)]*(self.nhistory_hour-1)
                    + [np.ptp(self.solar_arr)]*(self.nhistory_hour-1)
                    + [np.ptp(self.wind_arr)]*(self.nhistory_hour-1)
                )
                dt["demand_fcst_diffs"] = spaces.Box(
                    low=-np.ptp(self.demand_arr), 
                    high=np.ptp(self.demand_arr), 
                    shape=(self.nhistory-1,)
                )
                dt["solar_fcst_diffs"] = spaces.Box(
                    low=-np.ptp(self.solar_arr), 
                    high=np.ptp(self.solar_arr), 
                    shape=(self.nhistory-1,)
                )
                dt["wind_fcst_diffs"] = spaces.Box(
                    low=-np.ptp(self.wind_arr), 
                    high=np.ptp(self.wind_arr), 
                    shape=(self.nhistory-1,)
                )

        elif self.mode == Mode.SIGMOID_DIFF:
            highs = np.ones(self.nhistory)
            lows = 0 * highs
            dt = OrderedDict([
                ("battery_soc", spaces.Box(low=0, high=1, shape=())),
                ("lmp_diff_sigmoids", spaces.Box(low=0, high=1, shape=(self.nhistory,))),
            ])
            if self.more_data:
                dt["demand_fcst_diff_sigmoids"] = spaces.Box(low=0, high=1, shape=(self.nhistory_hour-1,))
                dt["solar_fcst_diff_sigmoids"] = spaces.Box(low=0, high=1, shape=(self.nhistory_hour-1,))
                dt["wind_fcst_diff_sigmoids"] = spaces.Box(low=0, high=1, shape=(self.nhistory_hour-1,))

        if self.solar_coloc:
            self.actual_solar_obs_index = len(lows)
            highs = np.append(highs, [np.max(self.actual_solar_arr)] * self.nhistory_hour)
            lows = np.append(lows, [np.min(self.actual_solar_arr)]* self.nhistory_hour)

            dt["solars"] = spaces.Box(
                low=np.min(self.actual_solar_arr), 
                high=np.max(self.actual_solar_arr), 
                shape=(self.nhistory_hour,)
            )

        # self.lows = lows
        # self.highs = highs
        # self.observation_space = spaces.spaces.Box(low=lows, high=highs, dtype=float)
        self.observation_space = spaces.Dict(dt)

    def setup_action_space(self):
        if self.mode == Mode.FINE_CONTROL:
            self.action_space = spaces.Discrete(11)
            self.action_to_direction = {
                0: -self.battery_power,
                1: -self.battery_power * 0.8,
                2: -self.battery_power * 0.6,
                3: -self.battery_power * 0.4,
                4: -self.battery_power * 0.2,
                5: 0,
                6: self.battery_power * 0.2,
                7: self.battery_power * 0.4,
                8: self.battery_power * 0.6,
                9: self.battery_power * 0.8,
                10: self.battery_power 
            }
        elif self.mode == Mode.LONG_CHARGE:
            self.action_space = spaces.Discrete(5)
            self.action_to_direction = {
                0: -self.battery_power,
                1: 0,
                2: self.battery_power,
                3: self.battery_power, # +1 day of charging
                4: self.battery_power, # +2 days of charging
            }
        else:
            self.action_space = spaces.Discrete(3)
            self.action_to_direction = {
                0: -self.battery_power, # discharge (make money)
                1: 0,              # do nothing
                2: self.battery_power   # charge (lose money)
            }
        # 31 Mar, 2024: Constance said to not enlarge action space and use 
        #               solar to augment our current decision (e.g., if sell,
        #               then also sell solar)
        # if self.solar_coloc:
        #     if self.mode in [Mode.FINE_CONTROL, Mode.LONG_CHARGE]:
        #         warnings.warn("Mode {self.mode} not supported with solar co-location, using default charging options")
        #     self.action_space = spaces.Discrete(5)
        #     self.action_to_direction = {
        #         0: -self.battery_power, # discharge (make money)
        #         1: 0,              # do nothing
        #         2: self.battery_power,   # charge (lose money)
        #         3: -self.battery_power,  # sell solar
        #         4: self.battery_power,   # charge battery with solar
        #     }

    def _get_obs(self):
        rt_idx = self.time_step 
        da_idx = int(self.time_step/4) 

        recent_lmps = self.lmp_arr.take(
            np.arange(rt_idx-self.nhistory+1,rt_idx+1), 
            mode="wrap")[::-1]
        obs = OrderedDict([
            ("battery_soc", np.array([self.battery_storage])),
            ("lmps", recent_lmps), 
        ])

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
            # obs = np.append(obs, recent_demand) 
            # obs = np.append(obs, recent_solar) 
            # obs = np.append(obs, recent_wind)

            obs["demand_fcsts"] = recent_demand
            obs["solar_fcsts"] = recent_solar
            obs["wind_fcsts"] = recent_wind

        if self.mode == Mode.LONG_CHARGE:
            # obs = np.append(obs, self.n_charges_left)
            obs["remaining_charges"] = self.n_charges_left
        elif self.mode in [Mode.DIFFERENCE, Mode.SIGMOID_DIFF]:
            # obs = np.array([self.battery_storage])
            obs = OrderedDict([("battery_soc", np.array([self.battery_storage]))])
            # obs = np.append(obs, np.ediff1d(recent_lmps))
            obs["lmp_diffs"] = np.ediff1d(recent_lmps)

            if self.more_data:
                # obs = np.append(obs, np.ediff1d(recent_demand))
                # obs = np.append(obs, np.ediff1d(recent_solar))
                # obs = np.append(obs, np.ediff1d(recent_wind))
                obs["demand_fcst_diffs"] = np.ediff1d(recent_demand)
                obs["solar_fcst_diffs"] = np.ediff1d(recent_solar)
                obs["wind_fcst_diffs"] = np.ediff1d(recent_wind)

                if self.mode == Mode.SIGMOID_DIFF:
                    # obs[0] /= self.battery_capacity
                    # obs[1:] = np.divide(1., 1. + np.exp(-obs[1:]))
                    obs["battery_soc"] /= self.battery_capacity
                    obs["lmp_diff_sigmoids"] = np.divide(1., 1+np.exp(-obs["lmp_diffs"]))
                    obs["demand_fcst_diff_sigmoids"] = np.divide(1., 1+np.exp(-obs["demand_fcst_diffs"]))
                    obs["solar_fcst_diff_sigmoids"] = np.divide(1., 1+np.exp(-obs["solar_fcst_diffs"]))
                    obs["wind_fcst_diff_sigmoids"] = np.divide(1., 1+np.exp(-obs["wind_fcst_diffs"]))

                    del obs["lmp_diffs"]
                    del obs["demand_fcst_diffs"]
                    del obs["solar_fcst_diffs"]
                    del obs["wind_fcst_diffs"]

        if self.solar_coloc:
            recent_actual_solar = self.actual_solar_arr.take(
                np.arange(da_idx-self.nhistory_hour+1,da_idx+1), 
                mode="wrap")[::-1]
            # obs = np.append(obs, recent_actual_solar)
            obs["solars"] = recent_actual_solar

        return obs

    def _get_info(self, solar_reward=0, grid_reward=0):
        return {
            "time_step": self.time_step,
            "solar_reward": solar_reward,
            "grid_reward": grid_reward,
            "soc": self.battery_storage,
            "curr_lmp": self.lmp_arr[self.time_step],
        }

    def step(self, action):
        rt_idx = self.time_step 
        da_idx = int(self.time_step/4) 

        # adjust action based on battery constraints 
        if self.mode == Mode.LONG_CHARGE and self.n_charges_left > 0:
            action = 2 
        max_charge = self.battery_storage >= self.battery_capacity 
        min_charge = self.battery_storage <= 0
        if (action in [2,4] and max_charge) or (action in [0,3] and min_charge):
            action = 1
        if self.mode == Mode.LONG_CHARGE:
            if action == 3: # TODO: Magic number
                self.n_charges_left = 1
            elif action == 4: # TODO: Magic number
                self.n_charges_left = 2
            else:
                self.n_charges_left = max(0, self.n_charges_left-1)

        # Finds power from grid (if applicable, solar) wrt battery capacity
        charge_power = self.action_to_direction[action] # unit: MW
        if self.solar_coloc:
            # solar charges in the same direction as action
            solar_power = self.actual_solar_arr[da_idx] # unit: MW
        else:
            solar_power = 0

        # charge
        if charge_power > 0:
            max_charge = min(self.battery_capacity-self.battery_storage, 
                             self.battery_power*self.rt_scale) # unit: MWh 
            solar_energy = min(solar_power*self.rt_scale, max_charge)
            energy_from_grid = min(charge_power*self.rt_scale, 
                                   max_charge-solar_energy)
            battery_charge = solar_energy + energy_from_grid
            self.battery_storage += battery_charge
            solar_surplus = solar_power*self.rt_scale - solar_energy
        # discharge
        elif charge_power <= 0:
            solar_energy = solar_power*self.rt_scale
            # charge_power <= 0
            battery_charge = max(-self.battery_storage, 
                                  charge_power*self.rt_scale) # unit: MWh
            energy_from_grid = battery_charge-solar_energy
            self.battery_storage += battery_charge
            solar_surplus = solar_energy

        curr_lmp = self.lmp_arr[rt_idx] # unit: $/MWh
        if self.time_step < self.end_index-1: 
            self.time_step += 1
        else:
            self.time_step = self.start_index

        solar_reward = curr_lmp * solar_surplus
        grid_reward = 0
        if self.delay_cost:
            if charge_power > 0:
                # fraction of energy from charging now
                alpha = battery_charge/(self.battery_storage)
                # fraction of energy bought from the grid
                beta = energy_from_grid/battery_charge

                self.avg_bought_lmp = (1.-alpha)*self.avg_bought_lmp + alpha*beta*curr_lmp # unit: $/MWh
            else:
                grid_reward = (curr_lmp - self.avg_bought_lmp) * (-battery_charge)
        else:
            # assuming lmp>=0, discharge (or lose energy) yields profits 
            if charge_power > 0:
                grid_reward = (-energy_from_grid) * curr_lmp
            else:
                grid_reward = (-battery_charge) * curr_lmp

        # every time step penalty (e.g., from investment costs)
        reward = solar_reward + grid_reward
        reward -= self.daily_cost

        observation = self._get_obs()
        info = self._get_info(solar_reward, grid_reward)

        if self.render_mode == "human":
            self.render_frame()

        terminated = False
        truncated = False

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.avg_bought_lmp = 0
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

