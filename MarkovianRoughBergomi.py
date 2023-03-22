import numpy as np
from scipy.special import gamma
import quadpy
import orthopy
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore', np.RankWarning)

# Gaussian quadrature rule
def exp_underflow(x):
    """
    computes exp(-x) avoiding underflow errors
    x: float of numpy array
    """
    if isinstance(x, np.ndarray):
        if x.dtype == int:
            x = x.astype(np.float)
        eps = np.finfo(x.dtype).tiny
    else:
        if isinstance(x, int):
            x = float(x)
        eps = np.finfo(x.__class__).tiny
    log_eps = -np.log(eps) / 2
    result = np.exp(-np.fmin(x, log_eps))
    result = np.where(x > log_eps, 0, result)
    return result


def settings(T, H, N):
    N = N - 1
    A = np.sqrt(1 / H + 1 / (1.5 - H))

    alpha = 1.06418
    beta = 0.4275

    z = np.exp(alpha * beta)
    factor_1 = (9 - (6 * H)) / (2 * H) ** (z / 8 * (z - 1))
    factor_2a = 5 * np.pi ** 3 / 768 * z * (z - 1) * (A ** (2 - 2 * H) * (3 - 2 * H)) / (
                beta ** (2 - 2 * H) * H) * N ** (1 - H)
    factor_2b = 5 * np.pi ** 3 / 1152 * z * (z - 1) * A ** (2 - 2 * H) / beta ** (2 - 2 * H) * N ** (1 - H)

    gamma_ = 1 / (3 * z / (8 * (z - 1)) + 6 * H - 4 * H ** 2)
    a = (1 / T) * (factor_1 * factor_2a ** (2 * H)) ** gamma_
    b = (1 / T) * (factor_1 * factor_2b ** (2 * H - 3)) ** gamma_

    m = int(np.fmax(np.round(beta / A * np.sqrt(N)), 1))
    n = int(np.round(N / m))

    # a is xi_0, b is xi_n
    sub_intervals = np.exp(np.log(a) + np.log(b / a) * np.linspace(0, 1, n + 1))  # ???
    return sub_intervals, m

def gauss_quadrature(H, m, a, b):
    '''
    returns the nodes and weights of the Gauss quadrature rule of level m on [a, b]
    '''
    d = np.arange(2 * m) + 0.5 - H
    c_H = 1 / (gamma(0.5 + H) * gamma(0.5 - H))
    alpha, beta, int_1 = orthopy.tools.chebyshev(moments = c_H / d * (b ** d - a ** d))
    return quadpy.tools.scheme_from_rc(alpha, beta, int_1)

def gauss_quadrature_sub_intervals(H, m, subs):
    '''
    returns the nodes and weights of the Gaussian quadrature rule of level m on a subinterval
    '''
    n = len(subs) - 1
    nodes = np.empty(m * n)
    weights = np.empty(m * n)
    for i in range(n):
        new_nodes, new_weights = gauss_quadrature(H, m, subs[i], subs[i + 1])
        nodes[m*i : m*(i + 1)] = new_nodes
        weights[m*i : m*(i + 1)] = new_weights
    return nodes, weights


def gaussian_quadrature_rule(T, H, N):
    sub_intervals, m = settings(T, H, N)
    n = len(sub_intervals) - 1

    if N == 1:
        w_0 = T ** (H - 0.5) / gamma(H + 1.5)
        nodes, weights = np.array([0.]), np.array([w_0])
    else:
        nodes, weights = np.zeros(m * n + 1), np.empty(m * n + 1)
        nodes[1:], weights[1:] = gauss_quadrature_sub_intervals(H, m, sub_intervals)
        weights[0] = (T ** (H + 0.5) / gamma(H + 1.5) - np.sum(
            weights[1:] / nodes[1:] * (1 - exp_underflow(nodes[1:] * T)))) / T

    return nodes, weights

# Markovian approximation of rBergomi
def covariance_matrix(nodes, T, time_steps):
    '''
    compute Cholesky decomposition of the covariance matrix
    of the Gaussian vector (W_dt, int_0^dt e^(-x_i(dt-s))dW_s)_i=1,..,N
    '''
    dt = T / time_steps
    inner_nodes = nodes[1:]
    sum_nodes = inner_nodes[:, None] + inner_nodes[None, :]
    N = len(nodes)
    cov_matrix = np.empty((N, N))
    cov_matrix[0, 0] = dt
    cov_matrix[0, 1:] = (1 - exp_underflow(dt * inner_nodes)) / inner_nodes
    cov_matrix[1:, 0] = cov_matrix[0, :1]
    cov_matrix[1:, 1:] = (1 - exp_underflow(dt * sum_nodes)) / sum_nodes

    # cholesky decomposition can only applied for positive definite matrix
    computed_chol = False
    sqrt_cov = None
    while not computed_chol:
        try:
            sqrt_cov = np.linalg.cholesky(cov_matrix)
            computed_chol = True
        except np.linalg.LinAlgError:
            dampening_factor = 0.999
            for i in range(1, N):
                cov_matrix[:i, i] = cov_matrix[:i, i] * dampening_factor ** ((i + 1) / N)
                cov_matrix[i, :i] = cov_matrix[i, :i] * dampening_factor ** ((i + 1) / N)
            computed_chol = False
    return sqrt_cov


def integral(nodes, weights, T, time_steps):
    '''
    compute the integral int_0^t (G_hat(t-s))^2 ds
    '''
    N = len(nodes)
    dt = T / time_steps
    t_n = np.arange(time_steps) * dt
    w_0 = weights[0]
    inner_nodes = nodes[1:]
    inner_weights = weights[1:]

    weight_mult = inner_weights[np.newaxis, :] * inner_weights[:, np.newaxis]
    node_sum = inner_nodes[np.newaxis, :] + inner_nodes[:, np.newaxis]

    integral = np.sum(
        (weight_mult / node_sum)[None, ...] * (1 - exp_underflow(node_sum[None, ...] * t_n[:, None, None])),
        axis=(1, 2))

    variance_w00 = w_0 ** 2 * t_n
    variance_w0i = w_0 * np.sum((inner_weights / inner_nodes)[np.newaxis, :] * (
                1 - exp_underflow(inner_nodes[np.newaxis, :] * t_n[:, np.newaxis])), axis=1)

    return integral + variance_w00 + 2 * variance_w0i  # np.array([t1 ... t_n ...])


def rBergomi_markovian(H, T, eta, rho, nodes, weights, time_steps, V0, S0, M):
    '''
    simulate M paths of stock prices by rough Bergomi model
    '''
    dt = T / time_steps
    N = len(nodes)
    sqrt_cov = covariance_matrix(nodes, T, time_steps)
    dW = np.einsum('ij,kjl->kil', sqrt_cov, np.random.normal(0, 1, size=(M, N, time_steps)))
    W = np.zeros((M, N, time_steps))

    for i in range(time_steps - 1):  # W is Brownian motion, then W_t2 = W_t1 + N(0, dt)
        W[:, :, i + 1] = exp_underflow(dt * nodes) * W[:, :, i] + dW[:, :, i]

    variance = integral(nodes, weights, T, time_steps)
    c = eta * np.sqrt(2 * H) * gamma(H + 0.5)
    W_int = np.einsum('ij,kjl->kl', weights[:, np.newaxis], W) * 1 / 3

    V = V0 * np.exp(c * W_int - 0.5 * c ** 2 * variance[np.newaxis, :])

    dWs = rho * dW[:, 0, :] + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), size=(M, time_steps))

    S = S0 * np.exp(np.sqrt(V) * dWs - 0.5 * V * dt)
    return S


def option_pricing_longstaff_schwartz(S0, stockprices, strike_price, deg, time_steps, M, T):
    dt = T / time_steps
    stockprices_transform = np.insert(stockprices, 0, np.full(M, S0), axis=1)
    payoffs = np.fmax(0, strike_price - stockprices_transform)
    itm = (payoffs > 0)

    # initiate stopping time and expected continuation value for all paths at expiration
    tau = np.full(M, time_steps)
    c = np.zeros(M)

    # backward iteration
    for t in range(time_steps, 0, -1):
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

    final_payoff = payoffs[np.arange(M), tau] * np.exp(-r * tau * dt)

    return np.average(final_payoff), 1.96 * np.std(final_payoff) / np.sqrt(M)
# test run
H = 0.01 # Hurst parameter of fractional Brownian motion
T = 1 # maturity
N = 6   # number of nodes for gaussian quadrature rule on each subinterval
eta = 1.9
rho = -0.9 # correlation coefficient of 2 Brownian motions driving stock price process and volatility process
nodes, weights = gaussian_quadrature_rule(T, H, N)
V0 = 0.235**2
S0 = 100
r = 0.02
strike_price = 105
deg = 3
time_steps = 50
M = 500000 # number of simulated paths
stockprices = rBergomi_markovian(H, T, eta, rho, nodes, weights, time_steps, V0, S0, M)
option_pricing_longstaff_schwartz(S0, stockprices, strike_price, deg, time_steps, M, T)

# sketch stockprice with N steps
M = 7
time_steps = 500
S0 = 1
N = 10
nodes, weights = gaussian_quadrature_rule(T, H, N)
stockprices = rBergomi_markovian(H, T, eta, rho, nodes, weights, time_steps, V0, S0, M)

x = np.linspace(0,T,time_steps+1)
stockprices_transform = np.insert(stockprices, 0, np.full(M, S0), axis=1)

fig = plt.figure(figsize=(10, 4))
for i in range(M):
    plt.plot(x, stockprices_transform[i], label = 'path'+str(i))

# Display a figure.
plt.show()
