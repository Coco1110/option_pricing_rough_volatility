import numpy as np
from scipy.special import gamma, hyp2f1, beta

def sqrt_cov_matrix(T, time_steps, H):
    """
    Return: Cholesky decomposition of the covariance matrix for the Gaussian vector
    Params: T: maturity
            H: Hurst parameter
    """
    N = time_steps
    dt = T / N
    cov = np.empty(shape=(2 * N, 2 * N))
    times = np.arange(1, N + 1) * dt
    minima = np.fmin(times[:, None], times[None, :])
    maxima = np.fmax(times[:, None], times[None, :])

    cov[0:N, 0:N] = (minima ** (0.5 + H)) * (maxima ** (H - 0.5)) * beta(1, H + 0.5) * hyp2f1(0.5 - H, 1, 1.5 + H, minima / maxima)
    cov[N:(2 * N), N:(2 * N)] = minima
    cov[0:N, N:(2 * N)] = 1 / (H + 0.5) * (times[:, None] ** (H + 0.5) - np.tril(maxima - minima) ** (H + 0.5))
    cov[N:(2 * N), 0:N] = cov[0:N, N:(2 * N)].T
    return np.linalg.cholesky(cov)


def rBergomi(H, T, time_steps, M, S0, V0, rho, eta):
    '''
    :param H: Hurst parameter
    :param T: Maturity
    :param time_steps: number of discrete time steps
    :param M: number of simulated paths
    :param S0: spot price
    :param V0: initial volatility
    :param rho: correlation
    :param eta:
    :return: stock price paths and volatility paths
    '''
    N = time_steps
    dt = T / N
    sqrt_cov = sqrt_cov_matrix(T, time_steps, H)
    W_int = sqrt_cov @ np.random.normal(0, 1, (2 * N, M))
    V = np.empty(shape=(N, M))  # actual V is of shape (N+1, M), but we do not need the last one for S
    V[0, :] = V0
    V[1:, :] = V0 * np.exp(
        eta * np.sqrt(2 * H) * W_int[:N - 1, :] - eta ** 2 / 2 * (np.arange(1, N) * dt)[:, None] ** (2 * H))

    dW = np.empty(shape=(N, M))
    dW[0, :] = W_int[N, :]
    dW[1:, :] = W_int[N + 1:, :] - W_int[N:2 * N - 1, :]

    Zt = rho * dW + np.random.normal(0, np.sqrt(1 - rho ** 2) * np.sqrt(dt), (N, M))
    S = S0 * np.exp(np.cumsum(np.sqrt(V) * Zt - V * dt / 2, axis=0))
    return np.insert(S, 0, np.full(M, S0), axis=0).T, np.insert(V, 0, np.full(M, V0), axis=0).T