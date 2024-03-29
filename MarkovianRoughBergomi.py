import numpy as np
from scipy.special import gamma
import orthopy
import quadpy

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
    '''
    Params: T: maturity
            H: Hurst parameter
            N: number of nodes used for the Guaussian quadrature rule
    return: partitions of the interval [a,b] and the level m of the Gaussian quadrature rule
    '''
    N = N - 1
    A = np.sqrt(1 / H + 1 / (1.5 - H))

    alpha = 1.06418
    beta = 0.4275

    # why float(N) instead of N???
    z = np.exp(alpha * beta)
    factor_1 = ((9 - 6 * H) / (2 * H)) ** (z / (8 * (z - 1)))
    factor_2a = 5 * np.pi ** 3 / 768 * z * (z - 1) * (A ** (2 - 2 * H) * (3 - 2 * H)) / (
                beta ** (2 - 2 * H) * H) * float(N) ** (1 - H)
    factor_2b = 5 * np.pi ** 3 / 1152 * z * (z - 1) * A ** (2 - 2 * H) / beta ** (2 - 2 * H) * float(N) ** (1 - H)

    gamma_ = 1 / (3 * z / (8 * (z - 1)) + 6 * H - 4 * H * H)
    a = (1 / T) * (factor_1 * factor_2a ** (2 * H)) ** gamma_
    b = (1 / T) * (factor_1 * factor_2b ** (2 * H - 3)) ** gamma_

    m = int(np.fmax(np.round(beta / A * np.sqrt(N)), 1))
    n = int(np.round(N / m))

    xi_0 = a * np.exp(-alpha * np.sqrt(N) / ((1.5 - H) * A))
    xi_n = b * np.exp(alpha * np.sqrt(N) / (H * A))
    sub_intervals = np.exp(np.log(xi_0) + np.log(xi_n / xi_0) * np.linspace(0, 1, n + 1))
    return sub_intervals, m

def gauss_quadrature(H, m, a, b):
    '''
    returns the nodes and weights of the Gauss quadrature rule of level m on [a, b]
    '''
    d = np.arange(2 * m) + 0.5 - H
    c_H = 1 / (gamma(0.5 + H) * gamma(0.5 - H))
    alpha, beta, int_1 = orthopy.tools.chebyshev(moments = c_H / d * (b ** d - a ** d))
    return quadpy.tools.scheme_from_rc(alpha, beta, int_1)

def gaussian_optimal_zero_weight(H, T, nodes, weights):
    """
    Computes the optimal weight in the L^2-approximation of an additional node at 0 given that we are already using the
    specified nodes and weights.
    :param H: Hurst parameter
    :param T: Final time
    :param nodes: a numpy array
    :param weights: a numpy array
    :return: The optimal weight in the L^2-sense of an additional node at 0
    """
    if len(nodes) == 0:
        return T ** (H - 0.5) / gamma(H + 1.5)
    return (T ** (H + 0.5) / gamma(H + 1.5) - np.sum(weights / nodes * (1 - exp_underflow(nodes * T)))) / T


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
    '''
    Params: T: maturity
            H: Hurst parameter
            N: number of nodes, weights used for the gaussian quadrature rule
    Return: all nodes and weights on all subintervals
    '''
    sub_intervals, m = settings(T, H, N)
    n = len(sub_intervals) - 1

    if N == 1:
        w_0 = T ** (H - 0.5) / gamma(H + 1.5)
        nodes, weights = np.array([0.]), np.array([w_0])
    else:
        nodes, weights = np.zeros(m * n + 1), np.empty(m * n + 1)
        nodes[1:], weights[1:] = gauss_quadrature_sub_intervals(H, m, sub_intervals)
        weights[0] = gaussian_optimal_zero_weight(H, T, nodes[1:], weights[1:])
    index_order = np.argsort(nodes)
    sorted_nodes = nodes[index_order]
    sorted_weights = weights[index_order]
    # when exist more than 1 zero nodes???
    if sorted_nodes[0] > 1e-04:
        nodes_ = np.zeros(len(sorted_nodes) + 1)
        nodes_[1:] = sorted_nodes
        weights_ = np.zeros(len(sorted_weights) + 1)
        weights_[1:] = sorted_weights
        sorted_nodes, sorted_weights = nodes_, weights_
    n_zero_nodes = np.sum(sorted_nodes < 1e-04)
    sorted_nodes[:n_zero_nodes] = 0.
    if n_zero_nodes > 1:
        nodes_ = np.zeros(len(sorted_nodes) - n_zero_nodes + 1)
        nodes_[1:] = sorted_nodes[n_zero_nodes:]
        weights_ = np.zeros(len(sorted_nodes) - n_zero_nodes + 1)
        weights_[0] = np.sum(sorted_weights[:n_zero_nodes])
        weights_[1:] = sorted_weights[n_zero_nodes:]
        sorted_nodes, sorted_weights = nodes_, weights_
    return sorted_nodes, sorted_weights

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
    entry = (1 - exp_underflow(dt * inner_nodes)) / inner_nodes
    cov_matrix[0, 0] = dt
    cov_matrix[0, 1:] = entry
    cov_matrix[1:, 0] = entry
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
    return integral + variance_w00 + 2 * variance_w0i
def rBergomi_markovian(H, T, time_steps, M, eta, rho, nodes, weights, V0, S0):
    '''
    simulate M paths of stock prices, volatilities by Markovian approximation of rough Bergomi model
    Params: H: Hurst parameter
            T: maturity
            eta: vol of vol
            rho: correlation of 2 Brownian motions
            nodes, weights: nodes and weights of Gaussian quadrature rule
            V0: forward variance constant
            S0: spot price
            M: number of simualted paths
    '''
    dt = T / time_steps
    N = len(nodes)
    sqrt_cov = covariance_matrix(nodes, T, time_steps)

    active_nodes = nodes
    active_weights = weights
    if weights[0] == 0:  # this is faster if the zero-node is not used in the volatility process
        active_nodes = nodes[1:]
        active_weights = weights[1:]

    dW = np.einsum('ij,kjl->kil', sqrt_cov, np.random.normal(0, 1, size=(M, N, time_steps)))
    W = np.zeros((M, N, time_steps))
    for i in range(time_steps - 1):  # W is Brownian motion, then W_t2 = W_t1 + N(0, dt)
        W[:, :, i + 1] = exp_underflow(dt * active_nodes) * W[:, :, i] + dW[:, :, i]

    variance = integral(active_nodes, active_weights, T, time_steps)
    c = eta * np.sqrt(2 * H) * gamma(H + 0.5)
    W_int = np.dot(c * active_weights, W)
    V = V0 * np.exp(W_int - 0.5 * c ** 2 * variance)
    dWs = rho * dW[:, 0, :] + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), size=(M, time_steps))
    S = S0 * np.exp(np.cumsum(np.sqrt(V) * dWs - 0.5 * V * dt, axis=1))
    return np.insert(S, 0, np.full(M, S0), axis=1), np.insert(V, 0, np.full(M, V0), axis=1)
