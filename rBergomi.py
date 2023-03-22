import numpy as np
from scipy.special import gamma, hyp2f1, beta
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore', np.RankWarning)


def sqrt_cov_matrix(T, N, H):
    """
    Construct the joint covariance matrix for the Volterra process W̃
    and the Brownian motion Z
    and compute its Cholesky decomposition.
    """
    dt = T / N
    cov = np.empty(shape=(2 * N, 2 * N))
    times = np.arange(1, N + 1) * dt
    minima = np.fmin(times[:, None], times[None, :])
    maxima = np.fmax(times[:, None], times[None, :])

    cov[0:N, 0:N] = (minima ** (0.5 + H)) * (maxima ** (H - 0.5)) * beta(1, H + 0.5) * hyp2f1(0.5 - H, 1, 1.5 + H,
                                                                                              minima / maxima)
    cov[N:(2 * N), N:(2 * N)] = minima
    cov[0:N, N:(2 * N)] = 1 / (H + 0.5) * (times[:, None] ** (H + 0.5) - np.tril(maxima - minima) ** (H + 0.5))
    cov[N:(2 * N), 0:N] = cov[0:N, N:(2 * N)].T

    return np.linalg.cholesky(cov)

def variance_process(H, T, N, eta, V0, m, sqrt_cov):
    dt = T / N
    W_int = sqrt_cov @ np.random.normal(0, 1, (2 * N, m))
    V = np.empty(shape=(N, m))  # actual V is of shape (N+1, M), but we do not need the last one for S
    V[0, :] = V0
    V[1:, :] = V0 * np.exp(eta * np.sqrt(2 * H) * W_int[:N - 1, :] - eta ** 2 / 2 * (np.arange(1, N) * dt)[:, None] ** (2 * H))
    return V

def price_process(H, T, N, S0, rho, m, sqrt_cov, V):
    dt = T / N
    W_int = sqrt_cov @ np.random.normal(0, 1, (2 * N, m))
    dW = np.empty(shape=(N, m))
    dW[0, :] = W_int[N, :]
    dW[1:, :] = W_int[N + 1:, :] - W_int[N:2 * N - 1, :]
    Zt = rho * dW + np.random.normal(0, np.sqrt(1 - rho ** 2) * np.sqrt(dt), (N, m))
    S = S0 * np.exp(np.sqrt(V) * Zt - V * dt / 2)
    return S

def option_pricing_longstaff_schwartz(S0, stockprices, strike_price, deg):
    dt = T / N
    stockprices_transform = np.insert(stockprices, 0, np.full(m, S0), axis=0).T
    payoffs = np.fmax(0, strike_price - stockprices_transform)
    itm = (payoffs > 0)

    # initiate stopping time and expected continuation value for all paths at expiration
    tau = np.full(m, N)
    c = np.zeros(m)

    # backward iteration
    for t in range(N, 0, -1):
        itm_paths = np.where(itm[:, t - 1] == True)[0]

        # discounted payoff
        discounted_h_itm = payoffs[itm_paths, tau[itm_paths]] * np.exp(-r * (tau[itm_paths] - t + 1) * dt)
        S_itm = stockprices_transform[itm_paths, t - 1]

        # regression
        beta = np.polyfit(S_itm, discounted_h_itm, deg)
        c[itm_paths] = np.polyval(beta, S_itm)

        h_itm = payoffs[itm_paths, t - 1]
        optimal_stopping_paths = itm_paths[np.where(h_itm > c[itm_paths])[0]]
        tau[optimal_stopping_paths] = t - 1

    final_payoff = payoffs[np.arange(m), tau] * np.exp(-r * tau * dt)

    return np.average(final_payoff), 1.96 * np.std(final_payoff) / np.sqrt(m)

T = 1
N = 50
m = 500000
r = 0.1
H = 0.1
eta = 1.9
rho = -0.9

sqrt_cov = sqrt_cov_matrix(T, N, H)
V = variance_process(H, T, N, eta, 0.235**2, m, sqrt_cov)
stockprices = price_process(H, T, N, 100, rho, m, sqrt_cov, V)
option_pricing_longstaff_schwartz(100, stockprices, 105, 3)

N = 500
m = 7
T = 1
H = 0.1
eta = 1.9
V0 = 0.02497
S0 = 1
rho = -0.9
matrix = sqrt_cov_matrix(T, N, H)
var = variance_process(H, T, N, eta, V0, m, matrix)
price = price_process(H, T, N, S0, rho, m, matrix, var)

# sketch stockprice with N steps
x = np.linspace(0,T,N+1)
stockprices_transform = np.insert(price, 0, np.full(m, S0), axis=0).T

for i in range(m):
    plt.plot(x, stockprices_transform[i], label = 'path'+str(i))
#plt.legend()
# Display a figure.
plt.show()