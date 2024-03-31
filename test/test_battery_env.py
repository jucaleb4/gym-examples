import unittest
import numpy as np

import gymnasium as gym
import gym_examples 

battery_capacity = 400
battery_power = 50
daily_cost = 200
rt_scale = 0.25 # since we buy at 15min increments and LMP is in MWh

class TestBatteryEnv(unittest.TestCase):
    """ Tests BatteryEnv with `Penalize` setting.  """

    def get_battery_env(self, 
            battery_capacity, 
            battery_power, 
            nhistory=16, 
            mode="", 
            more_data=False,
            delay_cost=False,
            solar_coloc=False,
            daily_cost=0,
            start_index=0,
            end_index=-1
        ): 
        data = "real" # type of data

        env = gym.make(
            "gym_examples/BatteryEnv-v0", 
            battery_capacity=battery_capacity,
            battery_power=battery_power,
            nhistory=nhistory, 
            data=data, 
            mode=mode,
            more_data=more_data,
            delay_cost=delay_cost,
            daily_cost=daily_cost,
            solar_coloc=solar_coloc,
            start_index=start_index,
            end_index=end_index
        )
        return env

    def test_start_with_empty_charge(self):
        env = self.get_battery_env(battery_capacity, battery_power)
        s, _ = env.reset()
        battery_lvl = s[0]

        self.assertEqual(battery_lvl, 0)

    def test_buy_null_sell(self):
        env = self.get_battery_env(battery_capacity, battery_power)
        s_0, _ = env.reset()
        s_1, r_1, _, _, _ = env.step(2) # buy
        s_2, r_2, _, _, _ = env.step(1) # null
        s_3, r_3, _, _, _ = env.step(0) # sell

        self.assertEqual(s_0[0], 0)
        self.assertEqual(s_1[0], rt_scale * battery_power)
        self.assertEqual(s_2[0], rt_scale * battery_power)
        self.assertEqual(s_3[0], 0)
        self.assertEqual(r_1, -rt_scale * battery_power * s_0[1])
        self.assertEqual(r_2, 0)
        self.assertEqual(r_3, rt_scale * battery_power * s_2[1])

    def test_buy_full_and_sell_empty(self):
        env = self.get_battery_env(battery_capacity, battery_power)
        s_0, _ = env.reset()
        s_1, r_1, _, _, _ = env.step(0) 

        self.assertEqual(s_0[0], 0)
        self.assertEqual(s_1[0], 0)
        self.assertEqual(r_1, 0)

        # buy until full
        for i in range(int(battery_capacity/(rt_scale * battery_power))):
            s_i, r_i, _, _, _ = env.step(2) 
            self.assertEqual(s_i[0], rt_scale * battery_power*(i+1))
            self.assertEqual(r_i, -rt_scale * battery_power*s_i[2])

        s_full, r_full, _, _, _ = env.step(2)
        self.assertEqual(s_full[0], battery_capacity)
        self.assertEqual(r_full, 0)

    def test_delay_cost(self):
        env = self.get_battery_env(battery_capacity, battery_power, delay_cost=True)
        s_0, _ = env.reset()
        s_1, r_1, _, _, _ = env.step(2) 
        s_2, r_2, _, _, _ = env.step(1) 
        s_3, r_3, _, _, _ = env.step(0) 

        self.assertEqual(s_0[0], 0)
        self.assertEqual(s_1[0], rt_scale * battery_power)
        self.assertEqual(s_2[0], rt_scale * battery_power)
        self.assertEqual(s_3[0], 0)
        self.assertEqual(r_1, 0)
        self.assertEqual(r_2, 0)
        self.assertEqual(r_3, rt_scale * battery_power*(s_2[1]-s_0[1]))

    def test_daily_cost(self):
        env = self.get_battery_env(battery_capacity, battery_power, 
                                   daily_cost=daily_cost)

        env.reset()
        # buy
        for t in range(int(battery_capacity/battery_power)):
            s_i, r_i, _, _, _ = env.step(2)
            self.assertEqual(r_i, -rt_scale * battery_power*s_i[2]-daily_cost)
        # null
        for t in range(int(battery_capacity/battery_power)):
            s_i, r_i, _, _, _ = env.step(1)
            self.assertEqual(r_i, -daily_cost)
        # sell
        for t in range(int(battery_capacity/battery_power)):
            s_i, r_i, _, _, _ = env.step(0)
            self.assertEqual(r_i, rt_scale*battery_power*s_i[2]-daily_cost)

    def test_daily_and_delay_cost(self):
        env = self.get_battery_env(
            battery_capacity, 
            battery_power, 
            delay_cost=True, 
            daily_cost=daily_cost,
        )

        s_0, _ = env.reset()
        bought_lmps = []
        for t in range(int(battery_capacity/battery_power)):
            s_i, r_i, _, _, _ = env.step(2)
            bought_lmps.append(s_i[2])
            self.assertEqual(r_i, -daily_cost)
        # null
        for t in range(int(battery_capacity/battery_power)):
            s_i, r_i, _, _, _ = env.step(1)
            self.assertEqual(r_i, -daily_cost)
        # sell
        avg_bought_lmps = np.mean(bought_lmps)
        for t in range(int(battery_capacity/battery_power)):
            s_i, r_i, _, _, _ = env.step(0)
            # floating point error in mean 
            self.assertAlmostEqual(r_i, rt_scale * battery_power*(s_i[2]-avg_bought_lmps)-daily_cost)

    def test_lmp_start_end_index_and_index_wrapping(self):
        pass

    def test_difference(self):
        """ See battery_env for meaning of `difference`. """
        nhistory=16
        env = self.get_battery_env(
            battery_capacity, 
            battery_power, 
            delay_cost=True, 
            daily_cost=daily_cost,
            more_data=True,
            nhistory=nhistory,
            # mode="difference"
        )

        # get two distinct snapshot of the state space
        s_0, _ = env.reset()
        for t in range(100):
            s_k, _, _, _, _ = env.step(t%3)

        diff_env = self.get_battery_env(
            battery_capacity, 
            battery_power, 
            delay_cost=True, 
            daily_cost=daily_cost,
            more_data=True,
            nhistory=nhistory,
            mode="difference"
        )
        s_diff_0, _ = diff_env.reset()
        for t in range(100):
            s_diff_k, _, _, _, _ = diff_env.step(t%3)

        self.assertEqual(s_0[0], s_diff_0[0])
        self.assertEqual(s_k[0], s_diff_k[0])
        self.assertTrue(np.allclose(np.ediff1d(s_0[1:1+nhistory]), s_diff_0[1:nhistory]))
        self.assertTrue(np.allclose(np.ediff1d(s_k[1:1+nhistory]), s_diff_k[1:nhistory]))

        env.reset()
        diff_env.reset()
        # sanity check daily and delay cost should still work
        bought_lmps = []
        for t in range(int(battery_capacity/battery_power)):
            s_i, _, _, _, _ = env.step(2)
            _, r_i, _, _, _ = diff_env.step(2)
            bought_lmps.append(s_i[2])
            self.assertEqual(r_i, -daily_cost)
        # null
        for t in range(int(battery_capacity/battery_power)):
            env.step(1)
            s_i, r_i, _, _, _ = diff_env.step(1)
            self.assertEqual(r_i, -daily_cost)
        # sell
        avg_bought_lmps = np.mean(bought_lmps)
        for t in range(int(battery_capacity/battery_power)):
            s_i, _, _, _, _ = env.step(0)
            _, r_i, _, _, _ = diff_env.step(0)
            self.assertAlmostEqual(r_i, rt_scale*battery_power*(s_i[2]-avg_bought_lmps)-daily_cost)

    def test_sigmoid(self):
        nhistory=16
        env = self.get_battery_env(
            battery_capacity, 
            battery_power, 
            delay_cost=True, 
            daily_cost=daily_cost,
            more_data=True,
            nhistory=nhistory,
            # mode="difference"
        )

        # get two distinct snapshot of the state space
        s_0, _ = env.reset()
        for t in range(100):
            s_k, _, _, _, _ = env.step(t%3)

        diff_env = self.get_battery_env(
            battery_capacity, 
            battery_power, 
            delay_cost=True, 
            daily_cost=daily_cost,
            more_data=True,
            nhistory=nhistory,
            mode="sigmoid"
        )
        s_diff_0, _ = diff_env.reset()
        for t in range(100):
            s_diff_k, _, _, _, _ = diff_env.step(t%3)

        sigmoid = lambda arr : np.reciprocal(1. + np.exp(-arr))
        self.assertEqual(s_0[0], s_diff_0[0])
        self.assertEqual(s_k[0], s_diff_k[0])
        self.assertTrue(np.allclose(sigmoid(np.ediff1d(s_0[1:1+nhistory])), s_diff_0[1:nhistory]))
        self.assertTrue(np.allclose(sigmoid(np.ediff1d(s_k[1:1+nhistory])), s_diff_k[1:nhistory]))

        env.reset()
        diff_env.reset()
        # sanity check daily and delay cost should still work
        bought_lmps = []
        for t in range(int(battery_capacity/battery_power)):
            s_i, _, _, _, _ = env.step(2)
            _, r_i, _, _, _ = diff_env.step(2)
            bought_lmps.append(s_i[2])
            self.assertEqual(r_i, -daily_cost)
        # null
        for t in range(int(battery_capacity/battery_power)):
            env.step(1)
            s_i, r_i, _, _, _ = diff_env.step(1)
            self.assertEqual(r_i, -daily_cost)
        # sell
        avg_bought_lmps = np.mean(bought_lmps)
        for t in range(int(battery_capacity/battery_power)):
            s_i, _, _, _, _ = env.step(0)
            _, r_i, _, _, _ = diff_env.step(0)
            self.assertAlmostEqual(r_i, rt_scale * battery_power*(s_i[2]-avg_bought_lmps)-daily_cost)

    def test_solar_coloc(self):
        """ 
        Solar co-location.
        """
        nhistory=16
        # indices
        nhistory_hour = 4
        # location of solar forecast
        solar_idx = 1+nhistory+nhistory_hour
        # location of actual solar
        actual_solar_idx = 1+nhistory+3*nhistory_hour

        env = self.get_battery_env(
            battery_capacity, 
            battery_power, 
            delay_cost=True, 
            daily_cost=daily_cost,
            more_data=True,
            nhistory=nhistory,
            solar_coloc=True,
            mode="default"
        )

        self.assertEqual(env.action_space.n, 3)

        # test buy, sell, null still work (daily cost is penalty to encourage
        # battery use, e.g., investment cost of battery divided over lifetime
        s_0, _ = env.reset()
        s_1, r_1, _, _, _ = env.step(2) # buy
        s_2, r_2, _, _, _ = env.step(1) # null
        s_3, r_3, _, _, _ = env.step(0) # sell

        self.assertEqual(s_0[0], 0)
        # check battery storage is correct
        actual_battery_energy = rt_scale*battery_power
        self.assertEqual(s_1[0], actual_battery_energy)
        self.assertEqual(s_2[0], actual_battery_energy)
        actual_battery_energy -= rt_scale*(battery_power)
        self.assertEqual(s_3[0], actual_battery_energy)
        self.assertEqual(r_1, -daily_cost)
        # TODO: Actually calaculte this
        self.assertAlmostEqual(r_2, 417.6419157072262)
        self.assertAlmostEqual(r_3, 685.8515965228625)

        # go through environment until we have 4 forecasted hours of solar
        while 1:
            s_0, _, _, _, _ = env.step(1) # null
            if np.all(s_0[solar_idx:nhistory_hour] > 0):
                break

        # TODO: Fix this later
        """
        s_1, r_1, _, _, _ = env.step(4) # charge with solar
        s_2, r_2, _, _, _ = env.step(2) # charge regularly
        charge_energy_1 = s_1[0] - s_0[0]
        charge_energy_2 = s_2[0] - s_1[0]

        # test selling from solar (ensure the amount of energy sold correct)
        s_3, r_3, _, _, _ = env.step(0) # sell normally
        s_4, r_4, _, _, _ = env.step(3) # sell solar
        s_5, r_5, _, _, _ = env.step(4) # sell normally
        discharge_energy_1 = s_3[0] - s_2[0]
        discharge_energy_2 = s_4[0] - s_3[0]
        discharge_energy_3 = s_5[0] - s_4[0]

        # start with empty battery
        self.assertEqual(s_0[0], 0)

        # test buying procedure from solar and regular
        self.assertGreater(charge_energy_1, 0) # we charged
        self.assertLess(charge_energy_1, battery_power) # we cannot charge more than transfer amount
        self.assertAlmostEqual(charge_energy_2, battery_power)
        self.assertEqual(r_1, -daily_cost)   
        self.assertEqual(r_2, -daily_cost)

        # test selling procedure
        self.assertEqual(r_3, -daily_cost + charge_energy_1 * (s_3[1]-0)) # we gained money since initial charging was free
        self.assertEqual(discharge_energy_1, -charge_energy_1)
        self.assertGreater(r_4, -daily_cost) # free money from selling solar
        self.assertEqual(discharge_energy_2, 0)
        self.assertGreater(r_5, -daily_cost + battery_power * (s_5[1] - s_2[1])) # normal selling
        self.assertEqual(discharge_energy_2, -charge_energy_2)
        """
