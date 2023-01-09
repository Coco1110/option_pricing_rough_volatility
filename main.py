import numpy as np
import datetime
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.simplefilter('ignore', np.RankWarning)

m = 500000
T = 1
N = 50
dt = T/N
S0 = 100
V0 = 0.02497
r = 0.02
sigma = 0.559
kappa = 1.22136
theta = 0.03
rho = -0.9
K = 105

def BS_stockprice(S0, r, sigma, T, N, m):
    #    function: simulate m samples of a Black-Scholes paths at (n+1) equidistant times
    #   parameters:
    #    - r: interest rate
    #   - sigma: volatility constant
    #    - T: termination
    #    - N: number of time steps
    #    - m: number of paths
    Wt = np.zeros([m, N+1])
    dt = T/N
    dW = np.random.normal(0, np.sqrt(dt), (m, N))
    Wt[:, 1:] = np.cumsum(dW, axis=1)
    return S0*np.exp(sigma*Wt + (r - 0.5 * sigma ** 2) * np.linspace(0, T, N + 1)[None, :])


def discrete_heston(S0, V0, r, sigma, kappa, theta, rho, N, T, m):
    #    function: use Euler method to discretize Heston model and return stock price and variance discrete process for m paths
    #    parameters:
    #    - m: number of paths
    #    - N: number of steps
    #    - T: maturity
    #    - S0: initial stock price
    #    - V0: initial volatility
    #    - theta: the long variance, or long-run average variance of the price; as t tends to infinity, the expected value of νt tends to θ.
    #    - rho: the correlation of the two Wiener processes.
    #    - kappa: the rate of mean reversion.
    #    - sigma: the volatility of the volatility, or 'vol of vol', which determines the variance of νt.
    dt = T/N
    S = np.empty([m, N+1])
    V = np.empty([m, N+1])
    S[:, 0] = S0
    V[:, 0] = V0
    for i in range(N):
        Z1 = np.random.normal(0, np.sqrt(dt), m)
        Z2 = np.random.normal(0, np.sqrt(dt), m)
        Zs = rho*Z1 + np.sqrt(1-rho**2)*Z2
        V[:, i+1] = V[:, i] + kappa * \
            (theta - V[:, i])*dt + sigma*np.sqrt(V[:, i])*Z1
        V[:, i+1] = np.fmax(V[:, i+1], 0)
        S[:, i+1] = S[:, i] + r*S[:, i]*dt + np.sqrt(V[:, i])*S[:, i]*Zs
        S[:, i+1] = np.abs(S[:, i+1])
    return S, V


def get_payoff(stockprice_paths, strike_price):
    return np.fmax(0, strike_price - stockprice_paths)


# Tsitsiklis - Van Roy

def option_pricing_simple(stock_prices, payoffs, deg):
    #    regression analysis to calculate price of a Bermuda put option
    #    V: option values
    #    C: expected continuation values

    # initiate
    V = payoffs[:, N]
    C = np.zeros(m)
    for t in range(N-1, 0, -1):
        reg_coeffs = np.polyfit(stock_prices[:, t], V, deg)
        C = np.polyval(reg_coeffs, stock_prices[:, t])
        V = np.fmax(payoffs[:, t], C)
    return np.average(V)


def get_option_values(stock_prices, payoffs, deg):
    #    get option value function at each time and expected continuation value if not exercising immediately
    # initiate
    V = np.zeros([m, N+1])
    C = np.zeros([m, N+1])

    # regression
    V[:, N] = payoffs[:, N]
    for t in range(N-1, -1, -1):
        reg_coeffs = np.polyfit(stock_prices[:, t], V[:, t+1], deg)
        C[:, t] = np.polyval(reg_coeffs, stock_prices[:, t]) * np.exp(-r * dt)
        V[:, t] = np.fmax(payoffs[:, t], C[:, t])
    return V, C


def option_pricing_tsitsiklis(V, C, payoffs):
    #    option pricing according to discounted option values from stoppng time
    stopping_rule = np.argmax(payoffs >= C, axis=1)
    return np.average(V[np.arange(m), stopping_rule] * np.exp(-r * stopping_rule * dt))


# Longstaff-Schwartz
def option_pricing_longstaff_bayer(stockprices, payoffs, deg):
    itm = (payoffs > 0)

    # initiate
    stopping_set = np.full(m, N)
    C = np.zeros([m, N+1])

    # iteration (backward in time)
    for t in range(N-1, -1, -1):
        itm_index = np.where(itm[:, t] == True)[0]
        h_itm_stopping = payoffs[itm_index, stopping_set[itm_index]
                                 ] * np.exp(-r * (stopping_set[itm_index]-t) * dt)
        h_itm = payoffs[itm_index, t]
        S_itm = stockprices[itm_index, t]

        # regression to get discounted expected continuation values
        coeffs = np.polyfit(S_itm, h_itm_stopping, deg)
        C[itm_index, t] = np.polyval(coeffs, S_itm)  # discount?

        if t == 0:
            C[itm_index, t] = np.average(h_itm_stopping)

        # nearly optimal stopping time
        update_optimal_stopping = np.where(h_itm > C[itm_index, t])[0]
        # stopping_set[update_optimal_stopping] = stopping_set[update_optimal_stopping] - 1
        stopping_set[update_optimal_stopping] = t
    return np.average(payoffs[np.arange(m), stopping_set] * np.exp(-r * stopping_set * dt))


def longstaff_schwartz_multivariate_polynomial_reg(stockprices, volatilities, payoffs, deg):
    itm = (payoffs > 0)
    N = stockprices.shape[1] - 1
    # initiate
    stopping_set = np.full(m, N)
    C = np.zeros([m, N+1])

    # iteration (backward in time)
    for t in range(N-1, -1, -1):
        itm_index = np.where(itm[:, t])[0]
        h_itm_stopping = payoffs[itm_index, stopping_set[itm_index]
                                 ] * np.exp(-r * (stopping_set[itm_index]-t) * dt)
        h_itm = payoffs[itm_index, t]
        S_itm = stockprices[itm_index, t]
        V_itm = volatilities[itm_index, t]
        X_itm = np.array([S_itm, V_itm]).T
        # create a new matrix consisting of all polynomial combinations
        poly_model = PolynomialFeatures(deg)
        # transform
        poly_X_itm = poly_model.fit_transform(X_itm)
        # fit the model
        # poly_model.fit(poly_X_itm, h_itm_stopping)
        # use linear regression as a base
        regression_model = LinearRegression()
        regression_model.fit(poly_X_itm, h_itm_stopping)
        # get expected continuation values
        C[itm_index, t] = regression_model.predict(poly_X_itm)

        # nearly optimal stopping time
        update_optimal_stopping = np.where(h_itm > C[itm_index, t])[0]
        # stopping_set[update_optimal_stopping] = stopping_set[update_optimal_stopping] - 1
        stopping_set[update_optimal_stopping] = t
    return np.average(payoffs[np.arange(m), stopping_set] * np.exp(-r * stopping_set * dt))

m = 500000
T = 1
N = 50
dt = T/N
S0 = 100
V0 = 0.02497
r = 0.02
sigma = 0.559
kappa = 1.22136
theta = 0.03
rho = -0.9
K = 105

