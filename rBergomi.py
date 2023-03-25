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

def variance_process(H, T, time_steps, eta, V0, M, sqrt_cov):
    N = time_steps
    dt = T / N
    W_int = sqrt_cov @ np.random.normal(0, 1, (2 * N, M))
    V = np.empty(shape=(N, M))  # actual V is of shape (N+1, M), but we do not need the last one for S
    V[0, :] = V0
    V[1:, :] = V0 * np.exp(eta * np.sqrt(2 * H) * W_int[:N - 1, :] - eta ** 2 / 2 * (np.arange(1, N) * dt)[:, None] ** (2 * H))
    return V

def price_process(T, time_steps, S0, rho, M, sqrt_cov, V):
    '''
    Params: T: maturity
            S0: initial stock price
            rho: correlation
            M: number of simulated paths
            sqrt_cov: Cholesky decomposition of the covariance matrix of the Gaussian vector
            V: variance process
    Return: M stock price processes 
    '''
    N = time_steps
    dt = T / N
    W_int = sqrt_cov @ np.random.normal(0, 1, (2 * N, M))
    dW = np.empty(shape=(N, M))
    dW[0, :] = W_int[N, :]
    dW[1:, :] = W_int[N + 1:, :] - W_int[N:2 * N - 1, :]
    Zt = rho * dW + np.random.normal(0, np.sqrt(1 - rho ** 2) * np.sqrt(dt), (N, M))
    S = S0 * np.exp(np.sqrt(V) * Zt - V * dt / 2)
    return np.insert(S, 0, np.full(M, S0), axis=0).T
