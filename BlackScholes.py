import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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


# sketch stockprice with N steps
m = 7
N = 500
T = 1
x = np.linspace(0, T, N + 1)

stockprices = BS_stockprice(1, 0.06, 0.2, T, N, m)
for i in range(m):
    plt.plot(x, stockprices[i], label='path' + str(i))
# plt.legend()
# Display a figure.
plt.show()


def get_payoff(stockprice_paths, strike_price):
    return np.fmax(0, strike_price - stockprice_paths)


# Tsitsiklis - Van Roy

def option_pricing_tsitsiklis_simple(stockprices, payoffs, deg):
    '''
    standard regression to estimate price of a Bermuda put option
    V: option values
    C: expected continuation values
    '''
    # initiate option value and continuation value (conditional expectation) at expiration
    v = payoffs[:, N]
    c = np.zeros(m)

    # backward iteration
    for t in range(N, 0, -1):
        beta = np.polyfit(stockprices[:, t - 1], v * np.exp(-r * T / N), deg)
        c = np.polyval(beta, stockprices[:, t - 1])
        v = np.fmax(payoffs[:, t - 1], c)

    return np.average(v)


def option_pricing_tsitsiklis_stopping(stockprices, payoffs, deg):
    '''
    standard regression to estimate price of a Bermuda put option
    by simulating the expected pay-off due to the nearly optimal stopping time
    V: option values
    C: expected continuation values
    '''
    # initiate option value and continuation value (conditional expectation) at expiration
    V = np.full((N + 1, m), payoffs[:, N])
    C = np.zeros([N + 1, m])
    # print(V,C)

    # backward iteration
    for t in range(N, 0, -1):
        beta = np.polyfit(stockprices[:, t - 1], V[t, :] * np.exp(-r * T / N), deg)  # future option value is discounted
        C[t - 1, :] = np.polyval(beta, stockprices[:, t - 1])
        V[t - 1, :] = np.fmax(payoffs[:, t - 1], C[t - 1, :])
        # print(t)
        # print('expected value', C)
        # print('payoffs', payoffs.T)
        # print('option values', V)

    # print('---')

    # nearly optimal stopping time
    v = np.zeros(m)
    for i in range(m):
        stopping = np.where(payoffs[i, :] - C[:, i] > 0)[0]
        if len(stopping) == 0:  # if payoffs nowhere greater than expected value, then not stopping and option value is v0
            v[i] = V[0, i]
        else:
            v[i] = V[stopping[0], i]  # the first time payoff greater than expected value
        # print('path', i)
        # print(stopping)
        # print(v)
    return np.average(v)

# Longstaff-Schwartz
def option_pricing_longstaff_schwartz(stockprices, payoffs, deg):
    dt = T / N
    itm = (payoffs > 0)

    # initiate stopping time and expected continuation value for all paths at expiration
    tau = np.full(m, N)
    c = np.zeros(m)

    # backward iteration
    for t in range(N, 0, -1):
        # print(t)
        # print('stopping_times', tau)
        # print('continuation' ,c)

        itm_paths = np.where(itm[:, t - 1] == True)[0]
        # print(itm_paths)

        # discounted payoff
        discounted_h_itm = payoffs[itm_paths, tau[itm_paths]] * np.exp(-r * (tau[itm_paths] - t + 1) * dt)
        S_itm = stockprices[itm_paths, t - 1]
        # print('future payoffs wrt stopping time', payoffs[itm_paths, tau[itm_paths]])
        # print('discounted fututre payoff', discounted_h_itm)
        # regression
        beta = np.polyfit(S_itm, discounted_h_itm, deg)
        c[itm_paths] = np.polyval(beta, S_itm)

        # print('expected continuation', c[itm_paths])
        h_itm = payoffs[itm_paths, t - 1]
        # print('current_payoff', h_itm)
        optimal_stopping_paths = itm_paths[np.where(h_itm > c[itm_paths])[0]]
        # print('stopping for paths ',optimal_stopping_paths)
        tau[optimal_stopping_paths] = t - 1
        # print('update_topping times' ,tau)
        # print('------')
    final_payoff = payoffs[np.arange(m), tau] * np.exp(-r * tau * dt)

    return np.average(final_payoff), 1.96 * np.std(final_payoff) / np.sqrt(m)  # Monte-Carlo Error

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


def longstaff_schwartz_double_run(payoffs2, stockprices, payoffs, deg):
    dt = T / N
    itm = (payoffs > 0)

    # initiate stopping time and expected continuation value for all paths at expiration
    tau = np.full(m, N)
    c = np.zeros(m)

    # backward iteration
    for t in range(N, 0, -1):
        itm_paths = np.where(itm[:, t - 1] == True)[0]

        # discounted payoff
        discounted_h_itm = payoffs[itm_paths, tau[itm_paths]] * np.exp(-r * (tau[itm_paths] - t + 1) * dt)
        S_itm = stockprices[itm_paths, t - 1]

        # regression
        beta = np.polyfit(S_itm, discounted_h_itm, deg)
        c[itm_paths] = np.polyval(beta, S_itm)

        h_itm = payoffs2[itm_paths, t - 1]

        optimal_stopping_paths = itm_paths[np.where(h_itm > c[itm_paths])[0]]
        tau[optimal_stopping_paths] = t - 1

    final_payoff = payoffs2[np.arange(m), tau] * np.exp(-r * tau * dt)
    print(final_payoff)
    return np.average(final_payoff), 1.96 * np.std(final_payoff) / np.sqrt(m)

print('completed')