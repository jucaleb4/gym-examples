import unittest
import numpy as np

import gymnasium as gym
import gym_examples 

battery_capacity = 400
transfer_rate = 50
daily_cost = 200

class TestBatteryEnv(unittest.TestCase):
    """ Tests BatteryEnv with `Penalize` setting.  """

    def get_battery_env(self, 
            battery_capacity, 
            transfer_rate, 
            nhistory=16, 
            mode="", 
            more_data=False,
            delay_cost=False,
            daily_cost=0,
            start_index=0,
            end_index=-1
        ): 
        data = "real" # type of data

        env = gym.make(
            "gym_examples/BatteryEnv-v0", 
            battery_capacity=battery_capacity,
            transfer_rate=transfer_rate,
            nhistory=nhistory, 
            data=data, 
            mode=mode,
            more_data=more_data,
            delay_cost=delay_cost,
            daily_cost=daily_cost,
            start_index=start_index,
            end_index=end_index
        )
        return env

    def test_start_with_empty_charge(self):
        env = self.get_battery_env(battery_capacity, transfer_rate)
        s, _ = env.reset()
        battery_lvl = s[0]

        self.assertEqual(battery_lvl, 0)

    def test_buy_null_sell(self):
        env = self.get_battery_env(battery_capacity, transfer_rate)
        s_0, _ = env.reset()
        s_1, r_1, _, _, _ = env.step(2) # buy
        s_2, r_2, _, _, _ = env.step(1) # null
        s_3, r_3, _, _, _ = env.step(0) # sell

        self.assertEqual(s_0[0], 0)
        self.assertEqual(s_1[0], transfer_rate)
        self.assertEqual(s_2[0], transfer_rate)
        self.assertEqual(s_3[0], 0)
        self.assertEqual(r_1, -transfer_rate * s_0[1])
        self.assertEqual(r_2, 0)
        self.assertEqual(r_3, transfer_rate * s_2[1])

    def test_buy_full_and_sell_empty(self):
        env = self.get_battery_env(battery_capacity, transfer_rate)
        s_0, _ = env.reset()
        s_1, r_1, _, _, _ = env.step(0) 

        self.assertEqual(s_0[0], 0)
        self.assertEqual(s_1[0], 0)
        self.assertEqual(r_1, 0)

        # buy until full
        for i in range(int(battery_capacity/transfer_rate)):
            s_i, r_i, _, _, _ = env.step(2) 
            self.assertEqual(s_i[0], transfer_rate*(i+1))
            self.assertEqual(r_i, -transfer_rate*s_i[2])

        s_full, r_full, _, _, _ = env.step(2)
        self.assertEqual(s_full[0], battery_capacity)
        self.assertEqual(r_full, 0)

    def test_delay_cost(self):
        env = self.get_battery_env(battery_capacity, transfer_rate, delay_cost=True)
        s_0, _ = env.reset()
        s_1, r_1, _, _, _ = env.step(2) 
        s_2, r_2, _, _, _ = env.step(1) 
        s_3, r_3, _, _, _ = env.step(0) 

        self.assertEqual(s_0[0], 0)
        self.assertEqual(s_1[0], transfer_rate)
        self.assertEqual(s_2[0], transfer_rate)
        self.assertEqual(s_3[0], 0)
        self.assertEqual(r_1, 0)
        self.assertEqual(r_2, 0)
        self.assertEqual(r_3, transfer_rate*(s_2[1]-s_0[1]))

    def test_daily_cost(self):
        env = self.get_battery_env(battery_capacity, transfer_rate, daily_cost=daily_cost)

        env.reset()
        # buy
        for t in range(int(battery_capacity/transfer_rate)):
            s_i, r_i, _, _, _ = env.step(2)
            self.assertEqual(r_i, -transfer_rate*s_i[2]-daily_cost)
        # null
        for t in range(int(battery_capacity/transfer_rate)):
            s_i, r_i, _, _, _ = env.step(1)
            self.assertEqual(r_i, -daily_cost)
        # sell
        for t in range(int(battery_capacity/transfer_rate)):
            s_i, r_i, _, _, _ = env.step(0)
            self.assertEqual(r_i, transfer_rate*s_i[2]-daily_cost)

    def test_daily_and_delay_cost(self):
        env = self.get_battery_env(
            battery_capacity, 
            transfer_rate, 
            delay_cost=True, 
            daily_cost=daily_cost,
        )

        s_0, _ = env.reset()
        bought_prices = []
        for t in range(int(battery_capacity/transfer_rate)):
            s_i, r_i, _, _, _ = env.step(2)
            bought_prices.append(s_i[2])
            self.assertEqual(r_i, -daily_cost)
        # null
        for t in range(int(battery_capacity/transfer_rate)):
            s_i, r_i, _, _, _ = env.step(1)
            self.assertEqual(r_i, -daily_cost)
        # sell
        for t in range(int(battery_capacity/transfer_rate)):
            s_i, r_i, _, _, _ = env.step(0)
            self.assertEqual(r_i, transfer_rate*(s_i[2]-bought_prices[t])-daily_cost)

    def test_lmp_start_end_index_and_index_wrapping(self):
        pass

    def test_difference(self):
        nhistory=16
        env = self.get_battery_env(
            battery_capacity, 
            transfer_rate, 
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
            transfer_rate, 
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
        bought_prices = []
        for t in range(int(battery_capacity/transfer_rate)):
            s_i, _, _, _, _ = env.step(2)
            _, r_i, _, _, _ = diff_env.step(2)
            bought_prices.append(s_i[2])
            self.assertEqual(r_i, -daily_cost)
        # null
        for t in range(int(battery_capacity/transfer_rate)):
            env.step(1)
            s_i, r_i, _, _, _ = diff_env.step(1)
            self.assertEqual(r_i, -daily_cost)
        # sell
        for t in range(int(battery_capacity/transfer_rate)):
            s_i, _, _, _, _ = env.step(0)
            _, r_i, _, _, _ = diff_env.step(0)
            self.assertEqual(r_i, transfer_rate*(s_i[2]-bought_prices[t])-daily_cost)

    def test_sigmoid(self):
        nhistory=16
        env = self.get_battery_env(
            battery_capacity, 
            transfer_rate, 
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
            transfer_rate, 
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
        bought_prices = []
        for t in range(int(battery_capacity/transfer_rate)):
            s_i, _, _, _, _ = env.step(2)
            _, r_i, _, _, _ = diff_env.step(2)
            bought_prices.append(s_i[2])
            self.assertEqual(r_i, -daily_cost)
        # null
        for t in range(int(battery_capacity/transfer_rate)):
            env.step(1)
            s_i, r_i, _, _, _ = diff_env.step(1)
            self.assertEqual(r_i, -daily_cost)
        # sell
        for t in range(int(battery_capacity/transfer_rate)):
            s_i, _, _, _, _ = env.step(0)
            _, r_i, _, _, _ = diff_env.step(0)
            self.assertEqual(r_i, transfer_rate*(s_i[2]-bought_prices[t])-daily_cost)
