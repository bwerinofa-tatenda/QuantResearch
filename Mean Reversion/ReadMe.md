# Mean Reversion

Mean reversion is based on the understanding that phenomena that tend to do exceptionally well in
one period do poorly in the next, this follows the reasoning that performance is randomly distributed around a `mean`.
If price series closely followed this, doing well as a trader would be relatively simple. All we need to do is to buy low
(when the price is below the mean) then wait for reversion to the mean price, and then sell at this higher price. 

But, most price series are not mean reverting, instead they are geometric random walks. The returns, not the prices, are 
the ones that usually randomly distribute around a mean of zero. Unfortunately, we cannot trade on the mean reversion of 
returns.

## Stationarity
Those few price series that are found to be `mean reverting` are called **stationary**. There aren't too many price series existing on exchanges that are stationary. Fortunately, we can create our own. This can be done by combining two or more individual price series that are not mean reverting into a portfolio whose net market value(price) is mean reverting. These price series are said to be cointegrated and we can use the Cointegrated Augmented Dickey Fuller test, to test for cointegration. Because of the ability to artificially create stationary portfolios there are numerous opportunities for mean reversion strategies.

### Mean Reversion and Stationarity
Mean reversion and stationarity are two equivalent ways of looking at the same type of price series, but these two ways give rise to two diff erent statistical tests for such series.The mathematical description of a mean-reverting price series is that the change of the price series in the next period is proportional to the difference between the mean price and the current price. This gives rise to the ADF test, which tests whether we can reject the null hypothesis that the proportionality constant is zero.


