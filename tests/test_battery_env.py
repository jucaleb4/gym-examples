import unittest

import gymnasium as gym
import gym_examples 

class TestBatteryEnvWithPenalty(unittest.TestCase):
    """ Tests BatteryEnv with `Penalize` setting.  """

    def get_battery_env(self): 
        nhistory = 10 # number of days to store
        data = "periodic" # type of data
        mode_str = "penalize"

        env = gym.make(
            "gym_examples/BatteryEnv-v0", 
            nhistory=nhistory, 
            data=data, 
            mode=mode_str
        )
        return env

    def test_start_with_empty_charge(self):
        env = self.get_battery_env()
        s, _ = env.reset()
        battery_lvl = s[0]

        self.assertEqual(battery_lvl, 0)

    def test_sell(self):
        env = self.get_battery_env()
        env.reset()
        s_1, r_1, _, _, _ = env.step(2) # buy
        s_2, r_2, _, _, _ = env.step(0) # sell
        prev_battery_level = s_1[0]
        curr_battery_level = s_2[0]

        # self.assertLess(r_1, 0)
        self.assertGreater(r_2, 0)
        self.assertLess(curr_battery_level, prev_battery_level)

    def test_neutral(self):
        env = self.get_battery_env()
        env.reset() 
        s_1, _, _, _, _ = env.step(2) # buy
        s_2, reward, _, _, _ = env.step(1) # do nothing
        prev_battery_level = s_1[0]
        curr_battery_level = s_2[0]

        self.assertEqual(reward, 0)
        self.assertEqual(curr_battery_level, prev_battery_level)

    def test_buy(self):
        env = self.get_battery_env()
        env.reset() 
        s_1, r_1, _, _, _ = env.step(2) # buy
        s_2, r_2, _, _, _ = env.step(2) # buy
        prev_battery_level = s_1[0]
        curr_battery_level = s_2[0]

        self.assertLess(r_1, 0)
        self.assertLess(r_2, 0)
        self.assertGreater(prev_battery_level, 0)
        self.assertGreater(curr_battery_level, prev_battery_level)

    def test_penalty_for_not_charging(self):
        env = self.get_battery_env()
        env.reset() 
        env.step(2) # buy
        s_1, r_1, _, _, _ = env.step(0) # sell
        s_2, r_2, _, _, _ = env.step(1) # nothing
        s_3, r_3, _, _, _ = env.step(1) # nothing
        s_4, r_4, _, _, _ = env.step(1) # nothing, penalty starts

        battery_lvl_1 = s_1[0]
        battery_lvl_2 = s_2[0]
        battery_lvl_3 = s_3[0]
        battery_lvl_4 = s_4[0]

        num_consecutive_idle_1 = s_1[-1]
        num_consecutive_idle_2 = s_2[-1]
        num_consecutive_idle_3 = s_3[-1]
        num_consecutive_idle_4 = s_4[-1]

        # check battery levels are always at zero
        self.assertEqual(battery_lvl_1, 0)
        self.assertEqual(battery_lvl_2, 0)
        self.assertEqual(battery_lvl_3, 0)
        self.assertEqual(battery_lvl_4, 0)

        # check no reward
        self.assertGreater(r_1, 0)
        self.assertEqual(r_2, 0)
        self.assertEqual(r_3, 0)
        self.assertLess(r_4, 0)

        # check we are keeping track of doing nothing
        self.assertEqual(num_consecutive_idle_1, 0)
        self.assertEqual(num_consecutive_idle_2, 1)
        self.assertEqual(num_consecutive_idle_3, 2)
        self.assertEqual(num_consecutive_idle_4, 3)

    def test_sell_on_empty_battery(self):
        env = self.get_battery_env()
        env.reset() 
        s_1, r_1, _, _, _ = env.step(2) # buy
        s_2, r_2, _, _, _ = env.step(0) # sell
        s_3, r_3, _, _, _ = env.step(0) # sell
        prev_battery_level = s_1[0]
        curr_battery_level = s_2[0]

        battery_lvl_1 = s_1[0]
        battery_lvl_2 = s_2[0]
        battery_lvl_3 = s_3[0]

        self.assertLess(battery_lvl_2, battery_lvl_1)
        self.assertEqual(battery_lvl_3, battery_lvl_2)

        self.assertGreater(r_2, 0)
        self.assertEqual(r_3, 0)

