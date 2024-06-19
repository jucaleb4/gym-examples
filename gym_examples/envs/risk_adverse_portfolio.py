import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GiniPortfolioEnv(gym.Env):
    """ Risk-adverse portfolio optimization.  """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, n=2, T=5, R=1, **kwargs):
        self.n = n # number of assets to trade
        self.T = T # memory len
        self.t = 0 # time index
        self.R = R # risk tolerance (wrt Gini's mean difference)
        self.rho = kwargs.get('rho', 0) # penalty scale

        # rewards come from distribution N(mu, sig^2)
        self.mu_s  = np.linspace(start=0, stop=10, num=n, endpoint=True)
        self.sig_s = self.mu_s

        space_dt = dict({})

        # history of returns (used to calculate Gini's mean difference)
        space_dt["past_returns"] = spaces.Box(low=-np.inf, high=np.inf, shape=(T,), dtype=np.float64)
        # current individual returns (to ensure full observation)
        space_dt["individual_returns"] = spaces.Box(low=-np.inf, high=np.inf, shape=(T,), dtype=np.float64)

        # current holdings. We assume item 0 is cash
        space_dt["holdings"] = spaces.Box(low=0, high=1, shape=(n,), dtype=np.float64)
        self.observation_space = spaces.Dict(space_dt)

        # most amount of quantity we can buy/sell for each product
        # self.transact_lim = np.reciprocal(np.arange(1,n+1).astype('float'))
        self.transact_lim = np.ones(n, dtype=float)
        self.action_space = spaces.Box(
            low=-self.transact_lim, 
            high=self.transact_lim, 
            shape=(n,),
            dtype=np.float64
        )

    def _get_obs(self):
        obs = dict({})
        obs['past_returns'] = self.past_returns
        obs['individual_returns'] = self.individual_returns
        obs['holdings'] = self.holdings

        return obs

    def _get_info(self):
        """ Returns Gini's (absoluate) mean difference and constraints """

        # Get Gini coefficient (for past decision)
        Gini_partial_next = 0.
        Gini_partial_last = 0.
        for t_1 in range(self.T):
            for t_2 in range(1,t_1):
                Gini_partial_next += 2*abs(self.past_returns[t_1] - self.past_returns[t_2])
            Gini_partial_last += 2*abs(self.past_returns[0] - self.past_returns[t_1])

        Gini_partial_next /= 2*self.T**2
        Gini_partial_last /= 2*self.T**2
        Gamma = Gini_partial_next + Gini_partial_last
        
        """
        We will build (A,b) where Ax<=b, where:
        x[0:n] = a (actions)
        x[n:n+T-1] = d_(t,T+1) for t=2,...,T, where d_{t,T}=|y_t-y_T|
        x[n+T-1] = excess_Gini

        Constraints for Gini's next iteration:
        constraints: d_(t,T+1) >= y_t-y_(T+1) and d_(t,T+1) >= y_(T+1)-y_t
                     for t=2,3,...,T
        where y_(T+1)=<holdings_(T+1), r_(T+1)> = <holdings_T + a_T, r_(T+1)>.
        Reformulate as 
        [
            <a_T,-r_(T+1)> - d_(t,T+1) <= -y_t + <holdings_T, r_(T+1)>
            <a_T, r_(T+1)> - d_(t,T+1) <=  y_t + <holdings_T,-r_(T+1)>
        ]
        """
        A = np.zeros((2*(self.T-1), self.n+self.T), dtype=float)
        for i in range(self.T-1):
            A[2*i,  0:self.n] =-self.individual_returns
            A[2*i+1,0:self.n] = self.individual_returns
            A[2*i,  self.n+i] =-1
            A[2*i+1,self.n+i] =-1

        partial_curr_returns = np.dot(self.holdings, self.individual_returns)
        b = np.zeros(A.shape[0], dtype=float)
        b[0::2] = -self.past_returns[1:] + partial_curr_returns
        b[1::2] =  self.past_returns[1:] - partial_curr_returns
        
        """
        Gini's: 0.5*sum_(t,t')=2^(T+1) |y_t-y_t'|/(T**2) <= bar(R) 
        Equivalently: 
        [
            0.5*sum_(t,t')=2^(T+1) |y_t-y_t'|/(T**2) + excess_Gini == bar(R)
            ||
            sum_(t=2)^(T) |y_t-y_(T+1)|/(T**2) + excess_Gini == bar(R) - Gini_partial_next 
            =
            sum_(t=2)^(T) d_(t,T+1)/(T**2) + excess_Gini == bar(R) - Gini_partial_next
        ]
        where Gini_partial_next:=sum(2=t<t'<=T) |y_t-y_t'|/(T**2)
        """
        new_row = np.append(
            np.zeros(self.n, dtype=float), 
            np.append(
                np.ones(self.T-1)/(self.T**2), 
                1 # Excess Gini
            )
        )
        # equality, so we need both positive and negaitve
        A = np.vstack((A, np.atleast_2d(new_row)))
        A = np.vstack((A, -np.atleast_2d(new_row)))
        b = np.append(b, self.R - Gini_partial_next)
        b = np.append(b, -b[-1])

        # Feasible: 1'a = 0 
        new_rows = np.zeros((2, A.shape[1]), dtype=float)
        new_rows[0,:self.n] = 1.
        new_rows[1,:self.n] = -1.
        A = np.vstack((A, new_rows))
        b = np.append(b, np.zeros(2, dtype=float))

        # Simple constraints for next iteration (projection)
        # x_t+a_t >= 0 and transaction limit
        lbs = -np.minimum(self.holdings, self.transact_lim) 
        ubs = self.transact_lim

        return dict({
            'Gini': float(Gamma),
            'Gini_partial_next': Gini_partial_next, # Gini measure from t=2,...T (missing T-1)
            'Gini_max': float(self.R), # max Gini risk
            'A': A.astype(np.float32),
            'b': b.astype(np.float32),
            'lbs': lbs.astype(np.float32),
            'ubs': ubs.astype(np.float32),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # previous returns (in ascending order based on time)
        self.past_returns = np.zeros(self.T, dtype=np.float64)
        # current holdings (holding 0 is cash, risk-free)
        self.holdings = np.zeros(self.n, dtype=np.float64); self.holdings[0]=1
        # next individual returns
        self.individual_returns = self.np_random.normal(self.mu_s, self.sig_s)

        return self._get_obs(), self._get_info()

    def simplex_projection(self, x):
        y = np.maximum(0, np.minimum(1, x))
        if np.sum(y) <= 1e-8:
            y = np.zeros(len(x), dtype=float)
            y[0] = 1.
        return y/np.sum(y)

    def get_penalty(self):
        """ Gets penalty rho/2 |max(0,Gini-R)|^2 """
        Gini = self._get_info()['Gini']
        return self.rho/2 * (max(0, Gini-self.R))**2

    def step(self, action):
        """ 
        Order of transaction:

        1. Make trades (new holdings)
        2. Get returns
        3. Update new return rates
        """
        # apply action (projection in case we do not lie in prob simplex)
        next_holdings = self.simplex_projection(self.holdings + action)
        transaction   = next_holdings - self.holdings
        self.holdings = next_holdings

        # get returns and transaction costs
        curr_return = np.dot(self.holdings, self.individual_returns)
        transaction_costs = 2*np.sum(np.log(1+np.abs(transaction)))

        # update historical returns
        self.past_returns[:-1] = self.past_returns[1:]
        self.past_returns[-1]  = curr_return

        # get reward
        penalty = self.get_penalty()
        reward = curr_return - transaction_costs - penalty

        # get next returns 
        self.individual_returns = self.np_random.normal(self.mu_s, self.sig_s)

        observation = self._get_obs()
        terminated = truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info
