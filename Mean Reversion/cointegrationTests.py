# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:58:16 2020

@author: Tatenda Bwerinofa
"""

import numpy as np
import pandas as pd

##ETFs provide a fertile ground for finding cointegrating price series—and thus good candidates for pair trading.
##Both Canadian and Australian economies are commodity based, so they seem likely to cointegrate

''' We assume the price series of EWA is contained in array x, and that of EWC in array y'''

df  = pd.read_csv('inputData_EWA_EWC.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.date ##remove HH:MM:SS
df.set_index('Date', inplace=True)

df.plot()
df.plot.scatter(y = 'EWA', x = 'EWC')

## We can first determine the optimal hedge ratio by running a linear regression fit between two price series 

import statsmodels.formula.api as sm
results = sm.ols(formula='EWC ~ EWA', data=df[['EWA', 'EWC']]).fit()
print(results.params)
hedgeRatio = results.params[1]
print('Hedge Ratio: {}'.format(hedgeRatio))

## Then plot the residual EWC-hedgeRatio*EWA
dz = np.array(df[['EWC']])-hedgeRatio*np.array(df[['EWA']])
dz = pd.DataFrame(dz)
dz.plot()
## As expected, the plot of the residual EWC-hedgeRatio*EWA looks very stationary.

# =============================================================================
# CADF Test 
# =============================================================================

## Assume the EWA to be independent and run the CADF test.
import statsmodels.tsa.stattools as st
coint_t, pvalue, crit_value = st.coint(df['EWA'], df['EWC'], maxlag=1)
print(' ')
print('CADF Test:')
print('t-stat: {}'.format(coint_t))
print('p-value: {}'.format(pvalue))
print('crit: {}'.format(crit_value))

## The test stat of -3.64 is less than the critical value of -3.34 at the 95% level. 
## We reject the null hypothesis that λ = 0, EWA and EWC are cointergrating with 95% certainty

# =============================================================================
# Johansen Test
# =============================================================================

## In order to test for cointegration of more than two variables, we need to use the Johansen test
## Price variable y(t) are actually vectors representing multiple price series, and the coefficients λ and α
# are actually matrices.

# ΔY(t) = ΛY(t − 1) + M + A1ΔY(t − 1) + … + Ak ΔY(t − k) + ∋t
# if Λ = 0, we do not have cointegration

## We denote rank of Λ as r, and the number of price series n.(rxn)
## The number of independent portfolios that can be formed by various linear combinations of the cointegrating
# price series is equal to r.
## The Johansen test will calculate r for us in two different ways, both based on eigenvector decomposition of
# Λ. 
# One test produces the so-called trace statistic, and other produces the eigen statistic

'''The Johansen test has 3 inputs:
        y - the input matrix with each column representing a price series
        det_order =0, to have a constant offset but not a constant drift term.
        k - the number of lags = 1
        '''

import statsmodels.tsa.vector_ar.vecm as vm
result = vm.coint_johansen(df[['EWA', 'EWC']].values, det_order=0, k_ar_diff=1)
print(" ")
print('Johansen Test:')
print("Trace Statistic:")
print(result.lr1)
print('Crit: 90%   95%    99%')
print(result.cvt)
print(" ")
print("Eigen Statistic:")
print(result.lr2)
print('Crit: 90%   95%    99%')
print(result.cvm)

## We see that for the Trace Statistic test, the hypothesis r = 0 is rejected at the 99% level, and r ≤ 1 is rejected
# at the 95 percent level. 
## The Eigen Statistic test concludes that hypothesis r = 0 is rejected at the 95 percent level, and r ≤ 1 is rejected
# at the 95 percent as well. 
## This means that from both tests, we conclude that there are two cointegrating relationships between EWA and EWC.

## We know that for the CADF if we switched EWA from independant to dependant we reach a different conclusion.
## With the Johansen Test we dont need to run two seperate regressions. The Johansen test, in other words, is independent
# of the order of the price series.

## 3 price series:
df = pd.read_csv('inputData_EWA_EWC_IGE.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.date ##remove HH:MM:SS
df.set_index('Date', inplace=True)

result = vm.coint_johansen(df.values, det_order=0, k_ar_diff=1)
print("Trace Statistic:")
print(result.lr1)
print('Crit: 90%   95%    99%')
print(result.cvt)
print(" ")
print("Eigen Statistic:")
print(result.lr2)
print('Crit: 90%   95%    99%')
print(result.cvm)

#The eigenvalues and eigenvectors are contained in the arrays results.eig and results.evec, respectively.
print(" ")
print('Eigenvalues: {}'.format(result.eig.reshape(3,1)))
print('Eigenvectors: {}'.format(result.evec))
## The eigen vector contains 3 different ways to form the stationary portforlio (for a test with 3 inputs)

## We should expect the first cointegrating relation to be the “strongest”; that is, have the shortest
# half-life for mean reversion (result.evec[:, 0])
## Naturally, we pick this eigenvector to form our stationary portfolio (the eigenvector determines the shares
# of each ETF), and we can find its half-life by the same method as before when we were dealing with a stationary
# price series.

## We now have to compute the T × 1 array yport, which represents the net market value (price) of the portfolio, 
# which is equal to the number of shares of each ETF multiplied by the share price of each ETF, then summed
# over all ETFs

yport = pd.DataFrame(np.dot(df.values, result.evec[:, 0])) 

# Find value of lambda and thus the half-life of mean reversion by linear regression fit:
ylag = yport.shift()
deltaY = yport - ylag
df2 = pd.concat([ylag, deltaY], axis=1)
df2.columns = ['ylag', 'deltaY']
regress_results = sm.ols(formula="deltaY ~ ylag", data=df2).fit() ##Fit can deal with NaN in the top row
print(regress_results.params)
lam = regress_results.params['ylag']

halflife = -np.log(2)/lam
print("Halflife: {} days".format(halflife))

## A halflife of 23 days makes this triplet a good candidate for a mean reversion strategy






