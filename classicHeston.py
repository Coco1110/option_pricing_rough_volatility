import numpy as np

import warnings
warnings.simplefilter('ignore', np.RankWarning)

def heston_model(S0, V0, r, sigma, kappa, theta, rho, N, T, m):
    '''
    function: use Euler method to discretize Heston model and return stock price and variance discrete process for m paths
    parameters:
    - m: number of paths
    - N: number of steps
    - T: maturity
    - S0: initial stock price
    - V0: initial volatility
    - theta: the long variance, or long-run average variance of the price; as t tends to infinity, the expected value of νt tends to θ.
    - rho: the correlation of the two Wiener processes.
    - kappa: the rate of mean reversion.
    - sigma: the volatility of the volatility, or 'vol of vol', which determines the variance of νt.
    '''

    dt = T / N
    S = np.empty([m, N + 1])
    V = np.empty([m, N + 1])
    S[:, 0] = S0
    V[:, 0] = V0
    for i in range(N):
        Z1 = np.random.normal(0, np.sqrt(dt), m)
        Z2 = np.random.normal(0, np.sqrt(dt), m)
        Zs = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2
        V[:, i + 1] = V[:, i] + kappa * (theta - V[:, i]) * dt + sigma * np.sqrt(V[:, i]) * Z1
        V[:, i + 1] = np.fmax(V[:, i + 1], 0)
        S[:, i + 1] = S[:, i] + r * S[:, i] * dt + np.sqrt(V[:, i]) * S[:, i] * Zs
        S[:, i + 1] = np.abs(S[:, i + 1])
    return S, V


# sketch stockprice with N steps
m = 10
N = 20
T = 1
x = np.linspace(0, T, N + 1)
price_paths = []

model = heston_model(S0, V0, r, sigma, kappa, theta, rho, N, T, m)
stockprices = model[0]
for i in range(m):
    plt.plot(x, stockprices[i], label='path' + str(i))
plt.legend()
# Display a figure.
plt.show()

def get_payoff(stockprice_paths, strike_price):
    return np.fmax(0, strike_price - stockprice_paths)


def option_pricing_longstaff_schwartz(stockprices, payoffs, deg):
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

        h_itm = payoffs[itm_paths, t - 1]
        optimal_stopping_paths = itm_paths[np.where(h_itm > c[itm_paths])[0]]
        tau[optimal_stopping_paths] = t - 1
    final_payoff = payoffs[np.arange(m), tau] * np.exp(-r * tau * dt)

    return np.average(final_payoff), 1.96 * np.std(final_payoff) / np.sqrt(m)

S0 = 100
V0 = 0.01
r = 0.1
sigma = 0.2054
kappa = 2
theta = 0.26967
rho = -0.01615
T = 1
N = 50
m = 1000000

stockprices = heston_model(S0, V0, r, sigma, kappa, theta, rho, N, T, m)[0]
payoffs = get_payoff(stockprices, 105)

option_pricing_longstaff_schwartz(stockprices, payoffs, 3)
