# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 08:10:10 2020

@author: Tatenda Bwerinofa
"""

# =============================================================================
# Mean Reversion Trading of calender spreads
# =============================================================================

## The log market value of a calendar spread portfolio with a long far contract
# and a short near contract is simply γ(T1 − T2), with T2 > T1. Since T1 and T2
# are fixed for a particular calendar spread, we can use the (hopefully) mean- 
# reverting γ to generate trading signals.(γ is the roll return)

## We assume that the price of the CL contracts is stored in a τ × M array cl, 
# (where τ is the number of trading days) and M is the number of contracts. We 
# compute γ in the same way as in Example 5.3, and store the resulting values 
# γ in a τ × 1 array gamma. 

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv('inputDataDaily_CL_20120502.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.date
df.set_index('Date', inplace=True)

#1. As a first step, we find the half-life of γ.
# -fitting gamma to the forward curve
gamma = np.full(df.shape[0], np.nan)
for t in range(df.shape[0]):
    idx = np.where(np.isfinite(df.iloc[t, :]))[0]
    idxDiff = np.array(list(set(idx[1:])-set(idx)))
    if ((len(idx)>=5) & (all(idxDiff[0:4]==1))):
        FT = df.iloc[t, idx[:5]]
        T = sm.add_constant(range(FT.shape[0]))
        model = sm.OLS(np.log(FT.values), T)
        res = model.fit()
        gamma[t] = -12*res.params[1]
        
results = adfuller(gamma[np.where(np.isfinite(gamma))], maxlag=1, regression='c', autolag=None)
print(results)

gamma = pd.DataFrame(gamma)
gamma.fillna('ffill')

gammaGood = gamma[gamma.notna().values]
gammalag = gammaGood.shift()
deltagamma = gammaGood - gammalag
deltagamma = deltagamma[1:]
gammalag = gammalag[1:]

X = sm.add_constant(gammalag)
model = sm.OLS(deltagamma, X)
res = model.fit()
halflife = -(np.log(2))/res.params[0]
print('Halflife: ', halflife)
## The halflife is 41 days, to apply the linear mean reversion strategy we set
# the lookback equal to halflife

lookback = int(halflife)
MA = gamma.rolling(lookback).mean()
MSTD =  gamma.rolling(lookback).std()
zScore = (gamma - MA)/MSTD

## We need to pick a pair of contracts, far and near, on each historical day, 
# based on three criteria:
# 1. The holding period for a pair of contracts is 3 months (61 trading days).
# 2. We roll forward to the next pair of contracts 10 days before the current 
#    near contract’s expiration.
# 3. The expiration dates of the near and far contracts are 1 year apart

positions = np.zeros(df.shape)
isExpireDate = np.isfinite(df) & ~np.isfinite(df.shift(-1))
holddays = 3*21
numDaysStart = holddays+10
numDaysEnd = 10
spreadMonth = 12

for c in range(0, df.shape[1]-spreadMonth):
    expireIdx = np.where(isExpireDate.iloc[:,c])[-1]
    if c==0:
        startIdx = max(0, expireIdx - numDaysStart)
        endIdx = expireIdx-numDaysEnd
    else:
        myStartIdx = endIdx+1
        myEndIdx =  expireIdx - numDaysEnd
        if (myEndIdx-myStartIdx) >= holddays:
            startIdx=myStartIdx
            endIdx=myEndIdx
        else:
            startIdx = np.Inf
    
    if (len(expireIdx) > 0) & (endIdx>startIdx):
        positions[startIdx[0]:endIdx[0], c] = -1
        positions[startIdx[0]:endIdx[0], c + spreadMonth] = 1
        
## Once we have picked those contracts, we assume initially that we will hold
# a long position in the far contract, and a short position in the near one, 
# subject to revisions later.
                
positions[zScore.isna().values.flatten(), :] = 0
zScore.fillna(-np.Inf, inplace=True)

## Finally, we apply the linear mean reversion strategy to determine the true
# positions and calculate the unlevered daily returns of the portfolio. 
# (The daily return is the daily P&L divided by 2 because we have two contracts.)

positions[zScore.values.flatten() > 0, :] = -positions[zScore.values.flatten() > 0, :]
positions = pd.DataFrame(positions)
pnl=np.sum((positions.shift().values)*(df.pct_change().values), axis=1) # daily P&L of the strategy
ret=pnl/np.sum(np.abs(positions.shift()), axis=1)
(np.cumprod(ret+1)-1).plot()
APR = np.prod(1+ret)**(252/len(ret))-1
Sharpe = np.sqrt(252)*np.mean(ret)/np.std(ret)
print('APR: {} \nSharpe: {}'.format(APR, Sharpe)) 


         

