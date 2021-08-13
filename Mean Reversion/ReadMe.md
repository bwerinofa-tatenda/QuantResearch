# Mean Reversion

Mean reversion is based on the understanding that phenomena that tend to do exceptionally well in
one period do poorly in the next, this follows the reasoning that performance is randomly distributed around a `mean`.
If price series closely followed this, doing well as a trader would be relatively simple. All we need to do is to buy low
(when the price is below the mean) then wait for reversion to the mean price, and then sell at this higher price. 

But, most price series are not mean reverting, instead they are geometric random walks. The returns, not the prices, are 
the ones that usually randomly distribute around a mean of zero. Unfortunately, we cannot trade on the mean reversion of 
returns.

## Stationarity
Those few price series that are found to be `mean reverting` are called **stationary**. There aren't too many price series existing on exchanges that are stationary. Fortunately, we can create our own.
