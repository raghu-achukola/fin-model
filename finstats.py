import numpy as np

def mean(x):
    return sum(x)/len(x)
def std(x,sample = True):
    avg = mean(x)
    return np.sqrt(sum([(val -avg)**2 for val in x])/(len(x)-int(sample)))
def skew (x):
    return moment(x,3,True)/std(x)**3
def kurtosis(x):
    return (moment(x,4,True)/(moment(x,2,True)**2))
def moment(x,n, mu =False, sig = False):
    l = len(x)
    avg = mean(x)
    sigma = std(x,sample =False)
    if not sig:
        return sum([(val-int(mu)*avg)**n for val in x ])/l
    else:
        return sum([((val-int(mu)*avg)/sigma)**n for val in x])/l
def realized_vol(returns):
    n = len(returns)
    ssq = sum([r**2 for r in returns])
    return 100*np.sqrt(252*ssq/n)
def daily_returns(prices):
    n = len(prices)-1
    return [np.log(prices[i+1]/prices[i]) for i in range(n)]
def edf(sample,bars):
    n = len(sample)
    return [sum([1 for point in sample if point<=bar])/n for bar in bars]
def normalize(data):
    return numpy.array(data - mean(data))/std(data)
