import numpy as np

def edgify(bins):
    return bins[:-1] + bins[:-1]/2 - bins[1:]/2

def cnan(x):
    return np.sum(np.isnan(x))


def cinf(x):
    return np.sum(np.isinf(x) | np.isneginf(x))

def rinf(x):
    x[np.isinf(x) | np.isneginf(x)] = 0
    return x

def rnan(x):
    x = rinf(x)
    x = np.nan_to_num(x)
    return x

def not_tests(tests, sample_length):
    o = np.arange(sample_length)
    
    return list(set(o)-set(tests))
