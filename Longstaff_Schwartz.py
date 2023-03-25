import numpy as np

def option_pricing_longstaff_schwartz(T, M, time_steps, r, stockprices, strike_price, deg):
    '''
    compute price of American put option using Longstaff-Schwartz method
    return: option price and Monte-Carlo error
    Params: T: maturity
            M: number of simulated paths
            r: interest rate
            deg: degree of regression polynomial
            stockprices: array of size (M, time_steps)
    '''
    dt = T / time_steps
    payoffs = np.fmax(0, strike_price - stockprices)
    itm = (payoffs > 0)
    # initiate stopping time and expected continuation value for all paths at expiration
    tau = np.full(M, time_steps)
    c = np.zeros(M)
    # backward iteration
    for t in range(time_steps, 0, -1):
        itm_paths = np.where(itm[:, t - 1] == True)[0]
        # discounted payoff
        discounted_h_itm = payoffs[itm_paths, tau[itm_paths]] * np.exp(-r * (tau[itm_paths] - t + 1) * dt)
        s_itm = stockprices[itm_paths, t - 1]
        # regression
        beta = np.polyfit(s_itm, discounted_h_itm, deg)
        c[itm_paths] = np.polyval(beta, s_itm)
        # optimal stopping rule
        h_itm = payoffs[itm_paths, t - 1]
        optimal_stopping_paths = itm_paths[np.where(h_itm > c[itm_paths])[0]]
        tau[optimal_stopping_paths] = t - 1
    final_payoff = payoffs[np.arange(M), tau] * np.exp(-r * tau * dt)
    return np.round(np.average(final_payoff), 2), 1.96 * np.std(final_payoff) / np.sqrt(M)

def longstaff_schwartz_double_run(T, M, time_steps, r, strike_price, payoffs2, stockprices, deg):
    dt = T / time_steps
    payoffs = np.fmax(0, strike_price - stockprices)
    itm = (payoffs > 0)
    # initiate stopping time and expected continuation value for all paths at expiration
    tau = np.full(M, time_steps)
    c = np.zeros(M)
    # backward iteration
    for t in range(time_steps, 0, -1):
        itm_paths = np.where(itm[:, t - 1] == True)[0]
        # discounted payoff
        discounted_h_itm = payoffs[itm_paths, tau[itm_paths]] * np.exp(-r * (tau[itm_paths] - t + 1) * dt)
        s_itm = stockprices[itm_paths, t - 1]
        # regression
        beta = np.polyfit(s_itm, discounted_h_itm, deg)
        c[itm_paths] = np.polyval(beta, s_itm)
        # optimal stopping
        h_itm = payoffs2[itm_paths, t - 1]
        optimal_stopping_paths = itm_paths[np.where(h_itm > c[itm_paths])[0]]
        tau[optimal_stopping_paths] = t - 1
    final_payoff = payoffs2[np.arange(M), tau] * np.exp(-r * tau * dt)
    return np.round(np.average(final_payoff), 2), 1.96 * np.std(final_payoff) / np.sqrt(M)