import LongstaffSchwartz as ls
import MarkovianRoughBergomi as mb
import rBergomi as rb
import BlackScholes as bs
import classicHeston as heston

S0 = 100
r = 0.05
sigma = 0.3
M = 100
time_steps = 100
stock_prices = bs.bs_stockprice(S0, r, sigma, 1, time_steps, M)
option_price = ls.option_pricing_longstaff_schwartz(1, M, time_steps, r, stock_prices, 110, 2)
print('put price is:', option_price[0], 'and Monte Carlo error is:', option_price[1])