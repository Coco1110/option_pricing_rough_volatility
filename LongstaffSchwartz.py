import numpy as np
import warnings
from sklearn.linear_model import LinearRegression

warnings.simplefilter('ignore', np.RankWarning)

def option_pricing_longstaff_schwartz(T, r, stockprices, strike_price, deg, basis):
    '''
    compute price of American put option using Longstaff-Schwartz method
    Return: option price and Monte-Carlo error
    Params: T: maturity
            M: number of simulated paths
            r: interest rate
            deg: degree of regression polynomial
            stockprices: array of size (M, time_steps)
    '''
    M = stockprices.shape[0]
    time_steps = stockprices.shape[1] - 1
    dt = T / time_steps
    payoffs = np.fmax(0, strike_price - stockprices)
    itm = (payoffs > 0)
    # initiate stopping time and expected continuation value for all paths at expiration
    tau = np.full(M, time_steps)
    c = np.zeros(M)
    # backward iteration
    for t in range(time_steps, 0, -1):
        itm_paths = np.where(itm[:, t - 1] == True)[0]
        if len(itm_paths) == 0:
            pass
        else:
            # discounted payoff
            discounted_h_itm = payoffs[itm_paths, tau[itm_paths]] * np.exp(-r * (tau[itm_paths] - t + 1) * dt)
            s_itm = stockprices[itm_paths, t - 1]
            # regression
            if basis == 'legendre':
                coeffs = np.polynomial.legendre.legfit(s_itm, discounted_h_itm, deg)
            if basis == 'hermite':
                coeffs = np.polynomial.hermite.hermfit(s_itm, discounted_h_itm, deg)
            if basis == 'chebyshev':
                coeffs = np.polynomial.chebyshev.chebfit(s_itm, discounted_h_itm, deg)
            else:
                coeffs = np.polyfit(s_itm, discounted_h_itm, deg)
            # get expected continuation
            c[itm_paths] = np.polyval(coeffs, s_itm)
            # optimal stopping rule
            h_itm = payoffs[itm_paths, t - 1]
            optimal_stopping_paths = itm_paths[np.where(h_itm > c[itm_paths])[0]]
            tau[optimal_stopping_paths] = t - 1
    final_payoff = payoffs[np.arange(M), tau] * np.exp(-r * tau * dt)
    mc_error = 1.96 * np.std(final_payoff) / np.sqrt(M)
    return np.round(np.average(final_payoff), 2), mc_error

def longstaff_schwartz_multivariate(T, stockprices, volatilities, strike, deg):
    time_steps = len(stockprices[0]) - 1
    M = len(stockprices)
    dt = T / time_steps
    payoffs = np.fmax(0, strike - stockprices)
    itm = (payoffs > 0)
    # initiate stopping time and expected continuation value for all paths at expiration
    tau = np.full(M, time_steps)
    c = np.zeros(M)
    # backward iteration
    for t in range(time_steps, 0, -1):
        itm_paths = np.where(itm[:, t - 1] == True)[0]
        # discounted payoff
        if len(itm_paths)==0:
            pass
        else:
            discounted_h_itm = payoffs[itm_paths, tau[itm_paths]] * np.exp(-r * (tau[itm_paths] - t + 1) * dt)
            s_itm = stockprices[itm_paths, t - 1]
            v_itm = volatilities[itm_paths, t - 1]

            no_itm = len(itm_paths)
            if deg == 'm1':
                X_1 = np.vstack((np.ones(no_itm), s_itm, v_itm, s_itm**2, s_itm*v_itm))
            if deg == 'm2':
                X_1 = np.vstack((np.ones(no_itm), s_itm, v_itm, s_itm**2))
            if deg == 'm3':
                X_1 = np.vstack((np.ones(no_itm), s_itm, v_itm, s_itm**2, s_itm*v_itm, s_itm**2 * v_itm))

            X = X_1.T #transform
            regModel = LinearRegression()
            regModel.fit(X, discounted_h_itm) # fit the model
            # get expected continuation values
            c[itm_paths] = regModel.predict(X)
            # optimal stopping rule
            h_itm = payoffs[itm_paths, t - 1]
            optimal_stopping_paths = itm_paths[np.where(h_itm > c[itm_paths])[0]]
            tau[optimal_stopping_paths] = t - 1
    final_payoff = payoffs[np.arange(M), tau] * np.exp(-r * tau * dt)
    mc_error = 1.96 * np.std(final_payoff) / np.sqrt(M)
    return np.round(np.average(final_payoff), 2), mc_error