# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 23:05:36 2020

@author: Tatenda Bwerinofa
"""

# =============================================================================
# Bollinger Bands with Kalman Filter
# =============================================================================

import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pandas_datareader.data as web

os.environ['ALPHAVANTAGE_API_KEY'] = '8OMTVGI7149G441M'

df0 = web.DataReader('USDCAD', 'av-daily', start=datetime(2007,1,1), end=datetime(2020,6,30))
df1 = web.DataReader('USDZAR', 'av-daily', start=datetime(2007,1,1), end=datetime(2020,6,30))

df = df0.append(df1)
df = pd.DataFrame({'USDCAD':df0['close'], 'USDZAR':df1['close']})
df.index = pd.to_datetime(df.index, format='%Y%m%d').dt.date
df.set_index('Date', inplace=True)

## Initialize the x as the observable model(EWA) and y as the observable(EWC)
x = df['USDCAD']
y = df['USDZAR']

## Augment x with ones to accomodate for the possible offset in the regression
# between x and y
x = np.array(ts.add_constant(x))[:, [1,0]]

## if delta=1 the estimated beta will wildly flactuate based on latest observation
# if delta=0 it allows no change(like normal OLS regression)
delta = 0.0001

## measurement prediction 
yhat = np.full(y.shape[0], np.nan)
e = yhat.copy()
Q = yhat.copy()

## We denote R(t|t) by P(t). Initialize R, P and beta
R = np.zeros((2,2))
P = R.copy()
beta = np.full((2, x.shape[0]), np.nan)
Vw = (delta/(1-delta))*np.eye(2)
Ve = 0.001

## Initialize beta with (:, 1) to zero
beta[:, 0] = 0    

## Given the initial beta, R (and P)
for t in range(len(y)):
    if t > 0:
        beta[:, t] = beta[:, t-1]
        R = P + Vw
        
    yhat[t] = np.dot(x[t, :], beta[:, t])
    # print('First yhat[t]: ', yhat[t])
    
    Q[t] = np.dot(np.dot(x[t, :], R), x[t, :].T) + Ve ##Measurement variance prediction
    # print('Q[t]: ', Q[t])
    
    ## Observe y(t)
    e[t] = y[t]-yhat[t] ## measurement prediction error
    # print('e(t)', e[t])
    # print('Second yhat[t]: ', yhat[t])
    
    K = np.dot(R, x[t, :].T)/Q[t]  ##Kalman gain
    # print(K)
    
    beta[:, t] = beta[:, t] + np.dot(K, e[t]) ##State Update
    # print(beta[:, t])
    
    P = R - np.dot(np.dot(K, x[t, :]), R) ##State covariance update
    # print(R)
    
plt.plot(beta[0, :])
plt.plot(beta[1, :])
plt.plot(e[2:])
plt.plot(np.sqrt(Q[2:]))

## Modified BB strategy
longsEntry = e < -np.sqrt(Q)
longsExit  = e > 0

shortsEntry = e > np.sqrt(Q)
shortsExit  = e < 0
    
numUnitsLong = np.zeros(longsEntry.shape)
numUnitsLong[:] = np.nan

numUnitsShort = np.zeros(shortsEntry.shape)
numUnitsShort[:] = np.nan

## The rest of the code is the same as the first BB strategy
numUnitsLong[0] = 0
numUnitsLong[longsEntry] = 1
numUnitsLong[longsExit] = 0
numUnitsLong = pd.DataFrame(numUnitsLong)
numUnitsLong.fillna(method='ffill', inplace=True)

numUnitsShort[0] = 0
numUnitsShort[shortsEntry] = -1
numUnitsShort[shortsExit] = 0
numUnitsShort = pd.DataFrame(numUnitsShort)
numUnitsShort.fillna(method='ffill', inplace=True)

numUnits = numUnitsLong + numUnitsShort
positions = pd.DataFrame(np.tile(numUnits.values, [1,2]) * ts.add_constant(-beta[0, :].T)[:, [1,0]]*df.values)  
pnl = np.sum((positions.shift().values)*(df.pct_change().values), axis=1)
ret = pnl/np.sum(np.abs(positions.shift()), axis=1)
(pd.DataFrame(np.cumprod(1+ret)-1)).plot() #Plotting the cumilative returns
APR = np.prod(1+ret)**(252/len(ret))-1
Sharpe = np.sqrt(252)*np.mean(ret)/np.std(ret)
print('APR: {} \nSharpe: {}'.format(APR, Sharpe)) 

    
    
        
        


