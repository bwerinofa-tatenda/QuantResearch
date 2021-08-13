from sklearn import linear_model
import numpy as np
import pandas as pd
from scipy import stats
from math import floor
from datetime import timedelta


class PairsTradingAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        
        self.SetStartDate(2015,1,1)
        self.SetEndDate(2020,1,1)
        self.SetCash(10000)
        self.numdays = 250  # set the length of training period
        tickers = ["AUDUSD", "NZDUSD"]
        self.symbols = [] #initialize symbols
        
        self.threshold = 1.
        for i in tickers: #add an instance of the data
            self.symbols.append(self.AddForex(i, Resolution.Hour, Market.Oanda)
        for i in self.symbols:
            i.hist_window = RollingWindow[TradeBar](self.numdays) 


    def OnData(self, data):
        #if the data is not AU or NU return none on the function
        if not (data.ContainsKey("AUDUSD") and data.ContainsKey("NZDUSD")): 
            return
        
        for symbol in self.symbols:
            symbol.hist_window.Add(data[symbol])
        #load the data
    
        price_x = pd.Series([float(i.Close) for i in self.symbols[0].hist_window], 
                             index = [i.Time for i in self.symbols[0].hist_window])
        #close prices of x                     
        price_y = pd.Series([float(i.Close) for i in self.symbols[1].hist_window], 
                            index = [i.Time for i in self.symbols[1].hist_window])
        #close prices of y
        
        if len(price_x) < 250: return
        #if the length of the price series is less than 250(the training period) return none
        
# =============================================================================
#         ##the next few lines of code refer to the method of calculating the entry, here they use spread
#         spread = self.regr(np.log(price_x), np.log(price_y))
#         mean = np.mean(spread)
#         std = np.std(spread)
#         ratio = floor(self.Portfolio[self.symbols[1]].Price - self.Portfolio[self.symbols[0]].Price)
#         ##up to here
# =============================================================================
        
        quantity = float(self.CalculateOrderQuantity(self.symbols[0],0.4)) 
        #calculates the units to buy/sell
        
# =============================================================================
#         #the next few lines of code are the entry condition
#         if spread[-1] > mean + self.threshold * std:
#             if not self.Portfolio[self.symbols[0]].Quantity > 0 and not self.Portfolio[self.symbols[0]].Quantity < 0:
#                 self.Sell(self.symbols[1], 100) 
#                 self.Buy(self.symbols[0],  ratio * 100)
#         
#         elif spread[-1] < mean - self.threshold * std:
#             if not self.Portfolio[self.symbols[0]].Quantity < 0 and not self.Portfolio[self.symbols[0]].Quantity > 0:
#                 self.Sell(self.symbols[0], 100)
#                 self.Buy(self.symbols[1], ratio * 100) 
#         ##up to here
# =============================================================================
        
# =============================================================================
#        This is the exit condition         
#         else:
#             self.Liquidate()
# 
# =============================================================================
        
# =============================================================================
#     #Calculation of the entry criteria spread which is the hedge ratio: intercept in the linear reg
#     def regr(self,x,y):
#         regr = linear_model.LinearRegression()
#         x_constant = np.column_stack([np.ones(len(x)), x])
#         regr.fit(x_constant, y)
#         beta = regr.coef_[0]
#         alpha = regr.intercept_
#         spread = y - x*beta - alpha
#         return spread
# =============================================================================
        

