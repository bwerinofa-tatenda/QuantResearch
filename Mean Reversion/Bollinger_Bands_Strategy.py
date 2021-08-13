# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:27:35 2020

@author: Tatenda Bwerinofa
"""

# =============================================================================
# Bollinger Band Strategy 
# =============================================================================

import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts
import statsmodels.formula.api as sm

df = pd.read_csv('inputData_GLD_USO.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.date
df.set_index('Date', inplace=True) 

lookback = 20
hedgeRatio = np.full(df.shape[0], np.nan) 
for x in np.arange(lookback, len(hedgeRatio)):
    regress_results = sm.ols(formula='USO~GLD', data=df[(x-lookback):x]).fit()
    hedgeRatio[x-1] = regress_results.params[1]
    
yport = np.sum(ts.add_constant(-hedgeRatio)[:, [1,0]]*df, axis=1)
yport.plot()
df2=df['USO']/df['GLD']



###BB Strategy
## We initialize the number of units of the unit portfolio on the long side, 
# numUnitsLong, a Tx1 array, and then set one of its values to 1 if we have
# a long entry signal, and to 0 if we have a long exit signal; and vice versa
# for the number of units on the short side.
## For those days that do not have any entry or exit signals, we use the
# fillMissingData function to carry forward the previous day’s units.
# fillMissingData starts with the second row of an array, and overwrites any 
# cell’s NaN value with the corresponding cell’s value in the previous row
# ffill for forward fill
## Once numUnitsLong and numUnitsShort are computed, we can combine them to 
# find the net number of units denoted by numUnits.

entryZscore = 1
exitZscore = 0 

MA = yport.rolling(lookback).mean()
MSTD = yport.rolling(lookback).std()
Zscore = (yport-MA)/MSTD

longsEntry = Zscore < -entryZscore
longsExit  = Zscore > -entryZscore

shortsEntry = Zscore > entryZscore
shortsExit  = Zscore < exitZscore

numUnitsLong = np.zeros(longsEntry.shape)
numUnitsLong[:] = np.nan

numUnitsShort = np.zeros(shortsEntry.shape)
numUnitsShort[:] = np.nan

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
positions = pd.DataFrame(np.tile(numUnits.values, [1,2]) * ts.add_constant(-hedgeRatio)[:, [1,0]]*df.values)  
pnl = np.sum((positions.shift().values)*(df.pct_change().values), axis=1)
ret = pnl/np.sum(np.abs(positions.shift()), axis=1)
(pd.DataFrame(np.cumprod(1+ret)-1)).plot() #Plotting the cumilative returns
APR = np.prod(1+ret)**(252/len(ret))-1
Sharpe = np.sqrt(252)*np.mean(ret)/np.std(ret)
print('APR: {} \nSharpe: {}'.format(APR, Sharpe)) 
#pd.DataFrame(Zscore).plot()
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(df['USO'], df['GLD'])

print(df.head())
