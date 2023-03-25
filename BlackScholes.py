import numpy as np
def bs_stockprice(S0, r, sigma, T, time_steps, M):
    '''
    Return: M stock price samples using Black Scholes model
    Params: T: maturity
            S0: spot price
            sigma: volatility constant
            M: number of simulated paths
            r: interest rate
    '''
    N = time_steps
    Wt = np.zeros([M, N+1])
    dt = T/N
    dW = np.random.normal(0, np.sqrt(dt), (M, N))
    Wt[:, 1:] = np.cumsum(dW, axis=1)
    return S0*np.exp(sigma*Wt + (r - 0.5 * sigma ** 2) * np.linspace(0, T, N + 1)[None, :])


