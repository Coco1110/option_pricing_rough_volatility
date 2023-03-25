import numpy as np
import warnings
warnings.simplefilter('ignore', np.RankWarning)
def heston_model(S0, V0, r, sigma, kappa, theta, rho, time_steps, T, M):
    '''
    use Euler method to discretize Heston model and return stock price and variance discrete process for M paths
    Params: T: maturity
            m: number of paths
            S0: spot price
            V0: initial volatility
            theta: the long variance, or long-run average variance of the price; as t tends to infinity, the expected value of νt tends to θ.
            rho: the correlation of the two Wiener processes.
            kappa: the rate of mean reversion.
            sigma: the volatility of the volatility, or 'vol of vol', which determines the variance of Vt.
    '''
    N = time_steps
    dt = T / N
    s = np.empty([M, N + 1])
    v = np.empty([M, N + 1])
    s[:, 0] = S0
    v[:, 0] = V0
    for i in range(N):
        Z1 = np.random.normal(0, np.sqrt(dt), M)
        Z2 = np.random.normal(0, np.sqrt(dt), M)
        Zs = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2
        v[:, i + 1] = v[:, i] + kappa * (theta - v[:, i]) * dt + sigma * np.sqrt(v[:, i]) * Z1
        v[:, i + 1] = np.fmax(v[:, i + 1], 0)
        s[:, i + 1] = s[:, i] + r * s[:, i] * dt + np.sqrt(v[:, i]) * s[:, i] * Zs
        s[:, i + 1] = np.abs(s[:, i + 1])
    return s