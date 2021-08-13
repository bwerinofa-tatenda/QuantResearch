# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 20:10:09 2020

@author: Tatenda Bwerinofa
"""
# =============================================================================
# Buy on Gap Model
# =============================================================================

## Select all stocks near the market open whose returns from their
# previous day’s lows to today’s opens are lower than one standard
# deviation. The standard deviation is computed using the daily closeto-
# close returns of the last 90 days. These are the stocks that “gapped
# down.”
## Narrow down this list of stocks by requiring their open prices to be
# higher than the 20-day moving average of the closing prices.
## Buy the 10 stocks within this list that have the lowest returns from their
# previous day’s lows. If the list has fewer than 10 stocks, then buy the
# entire list.
## Liquidate all positions at the market close.


import numpy as np
import pandas as pd

cl = pd.read_csv('inputDataOHLCDaily_20120424_cl.csv')
op = pd.read_csv('inputDataOHLCDaily_20120424_op.csv')
hi = pd.read_csv('inputDataOHLCDaily_20120424_hi.csv')
lo = pd.read_csv('inputDataOHLCDaily_20120424_lo.csv')

stocks = pd.read_csv('inputDataOHLCDaily_20120424_stocks.csv')

cl['Var1'] = pd.to_datetime(cl['Var1'], format='%Y%m%d').dt.date
cl.columns = np.insert(stocks.values, 0, 'Date')
cl.set_index('Date', inplace=True)

op['Var1'] = pd.to_datetime(op['Var1'], format='%Y%m%d').dt.date
op.columns = np.insert(stocks.values, 0, 'Date')
op.set_index('Date', inplace=True)

hi['Var1'] = pd.to_datetime(hi['Var1'], format='%Y%m%d').dt.date
hi.columns = np.insert(stocks.values, 0, 'Date')
hi.set_index('Date', inplace=True)

lo['Var1'] = pd.to_datetime(lo['Var1'], format='%Y%m%d').dt.date
lo.columns = np.insert(stocks.values, 0, 'Date')
lo.set_index('Date', inplace=True)

topN = 10
entryZscore = 1
lookback = 20

stdc2c90d = cl.pct_change().rolling(90).std().shift(1)
buyPrice = lo.shift()*(1-entryZscore*stdc2c90d)

retGap = (op-lo.shift())/lo.shift()
ma = cl.rolling(lookback).mean()
positionsTable = np.zeros(retGap.shape) 

for t in np.arange(1, cl.shape[0]):
    hasData = np.where(np.isfinite(retGap.iloc[t, :]) & (op.iloc[t, :] < buyPrice.iloc[t, :]).values & (op.iloc[t, :] > ma.iloc[t, :]).values)
#    hasData=np.where(np.isfinite(retGap.iloc[t, :]) & (op.iloc[t, :] < buyPrice.iloc[t, :]).values & (op.iloc[t, :] > ma.iloc[t, :]).values)
    hasData = hasData[0]
    if len(hasData)>0:
        idxSort = np.argsort(retGap.iloc[t, hasData])
        positionsTable[t, hasData[idxSort.values[np.arange(-np.min((topN, len(idxSort))), 0)]]] = 1

retO2C = (cl-op)/op

pnl = np.sum(positionsTable*retO2C, axis=1)
ret = pnl/topN
(np.cumprod(1+ret)-1).plot()
APR = np.prod(1+ret)**(252/len(ret))-1
Sharpe = np.sqrt(252)*np.mean(ret)/np.std(ret)
print('APR: {} \nSharpe: {}'.format(APR, Sharpe))

## Experiment with increasing the lookback period(moving average)
# Experiment with making the model a short on gap instead 
    
    
    
    
    
    
    
    
    


