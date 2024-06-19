# python -m unittest  test.test_risk_adverse_portfolio

import unittest
import numpy as np
import cvxpy as cp
import numpy.linalg as la

import gymnasium as gym
import gym_examples 

class TestRiskAdversePortfolioEnv(unittest.TestCase):
    n = 4
    T = 4
    R = np.sqrt(n)
    
    def get_portfolio_env(rho=0): 
        return gym.make(
            "gym_examples/GiniPortfolioEnv-v0", 
            n=TestRiskAdversePortfolioEnv.n, 
            T=TestRiskAdversePortfolioEnv.T, 
            R=TestRiskAdversePortfolioEnv.R,
            rho=rho,
            max_episode_steps=1000,
        )

    def test_reset(self):
        """ Check reset properly resets all items """
        env = TestRiskAdversePortfolioEnv.get_portfolio_env()
        env.reset()
        for _ in range(100):
            env.step(env.action_space.sample())
        obs, info = env.reset()

        self.assertEqual(la.norm(obs['past_returns']), 0)
        # returns can be anything...
        self.assertEqual(obs['individual_returns'][0], 0)
        self.assertEqual(obs['holdings'][0], 1)
        self.assertEqual(la.norm(obs['holdings'][1:]), 0)

    def test_holdings(self):
        """ Test holdings are accurate after making some trades """
        env = TestRiskAdversePortfolioEnv.get_portfolio_env()
        n = TestRiskAdversePortfolioEnv.n

        obs, _ = env.reset()
        self.assertEqual(obs['holdings'][0], 1)
        self.assertEqual(la.norm(obs['holdings']), 1)

        action = np.zeros(n, dtype=float)
        # Transfer sequentially from (i-1)->i
        for i in range(1,n):
            action[i-1] = -1; action[i] = 1

            obs = env.step(action)[0]

            self.assertEqual(obs['holdings'][i], 1)
            self.assertEqual(la.norm(obs['holdings']), 1)

            action[i-1] = 0

        # Spread evenly
        action[:] = 1./n
        action[-1] -= 1
        obs = env.step(action)[0]
        self.assertTrue(np.allclose(obs['holdings'], 1./n*np.ones(n, dtype=float)))

        # Send everything back to asset 0 (cash)
        action[:] = -1./n
        action[0] += 1
        obs = env.step(action)[0]
        self.assertEqual(obs['holdings'][0], 1)
        self.assertEqual(la.norm(obs['holdings']), 1)

        # Test withdrawing from empty assets does not chnage
        init_holding = obs['holdings']
        for i in range(1,n):
            # transfer from (i)->0
            action[0] = 1; action[i] = -1

            obs = env.step(action)[0]

            self.assertTrue(np.allclose(init_holding, obs['holdings']))
            action[i] = 0

    def test_asset_zero_always_zero(self):
        """ Ensures asset 0 is always zero price """
        env = TestRiskAdversePortfolioEnv.get_portfolio_env()
        obs, _ = env.reset()

        for _ in range(100):
            obs = env.step(env.action_space.sample())[0]
            self.assertEqual(obs['individual_returns'][0], 0)

    def test_returns(self):
        env = TestRiskAdversePortfolioEnv.get_portfolio_env()
        n = TestRiskAdversePortfolioEnv.n
        T = TestRiskAdversePortfolioEnv.T
        obs, _ = env.reset()

        # randomly make trades and check past returns are correctly remembered
        past_returns = np.copy(obs['past_returns'])
        for _ in range(100):
            obs = env.step(env.action_space.sample())[0]
            curr_returns = np.copy(obs['past_returns'])
            self.assertTrue(np.allclose(past_returns[1:], curr_returns[:-1]))
            past_returns = curr_returns

        # check return is correct without making trades
        for _ in range(100):
            holdings = obs['holdings']
            individual_returns = obs['individual_returns']
            expected_return = np.dot(holdings, individual_returns)
            obs = env.step(np.zeros(n, dtype=float))[0]
            self.assertAlmostEqual(obs['past_returns'][-1], expected_return)

    def test_projection_bounds(self):
        env = TestRiskAdversePortfolioEnv.get_portfolio_env()
        n = TestRiskAdversePortfolioEnv.n
        T = TestRiskAdversePortfolioEnv.T
        obs, _ = env.reset()

        # randomly make trades and check bounds are correct
        past_returns = np.copy(obs['past_returns'])
        for _ in range(100):
            obs, _, _, _, info = env.step(env.action_space.sample())
            x = obs['holdings']
            lbs = info['lbs']; ubs = info['ubs']

            # we cannot sell more than holdings
            self.assertTrue(np.allclose(-lbs, x))

            # we cannot buy more than one
            self.assertTrue(np.allclose(ubs, 1))

    def test_valid_ginis(self):
        env = TestRiskAdversePortfolioEnv.get_portfolio_env()
        n = TestRiskAdversePortfolioEnv.n
        T = TestRiskAdversePortfolioEnv.T
        env.reset()

        # randomly make trades 
        for _ in range(np.random.randint(1,1000)):
            obs, _, _, _, info = env.step(env.action_space.sample())

        Gini_partial_next = info['Gini_partial_next']
        A,b = info['A'], info['b']
        # get feasible action that distributes everything evenly
        action = 1./n - obs['holdings']

        # calculate next Gini with a fixed action by solving LP
        # We want to minimize excess Gini (x[-1])
        x = cp.Variable(n+T)
        prob = cp.Problem(cp.Minimize(x[-1]), [A @ x <= b, x[:n] == action])
        prob.solve()

        self.assertEqual(prob.status, 'optimal')
        # remove Gini excess
        excess_gini = x.value[-1]
        # Gini contribution wrt return y_(T+1)
        Gini_partial = np.dot(A[-4], np.array(x.value)) + excess_gini
        expected_Gini = Gini_partial + Gini_partial_next

        # apply next action 
        obs = env.step(action)[0]
        self.assertAlmostEqual(expected_Gini, obs['Gini'])

    def test_penalty_f_and_df(self):
        """ Check the penalty function and its derivatve for excess Gini """
        rho = 1+999*np.random.random()
        env = TestRiskAdversePortfolioEnv.get_portfolio_env(rho)
        obs, info = env.reset()

        A = info['A']
        penalty_f = info['penalty_f']
        penalty_df= info['penalty_df']
        x = np.zeros(A.shape[1], dtype=float)

        # negative excess Gini
        x[-1] = -100*np.random.random() + TestRiskAdversePortfolioEnv.R
        f = penalty_f(x)
        df = penalty_df(x)
        self.assertEqual(f, 0)
        self.assertEqual(df[-1], 0)
        self.assertAlmostEqual(la.norm(df), 0)

        # no excess Gini
        x[-1] = TestRiskAdversePortfolioEnv.R
        f = penalty_f(x)
        df = penalty_df(x)
        self.assertEqual(f, 0)
        self.assertEqual(df[-1], 0)
        self.assertAlmostEqual(la.norm(df), 0)

        # excess Gini
        x[-1] = 100*np.random.random() + TestRiskAdversePortfolioEnv.R
        f = penalty_f(x)
        df = penalty_df(x)
        self.assertAlmostEqual(f, rho * (x[-1] - TestRiskAdversePortfolioEnv.R))
        self.assertEqual(df[-1], rho)
        self.assertAlmostEqual(la.norm(df), rho)
