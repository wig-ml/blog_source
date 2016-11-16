import numpy as np
import pandas as pd
import json, os, operator, heapq

base = os.path.dirname(os.getcwd())
data_stock = os.path.join(base, 'simple neural network')
directories = [x for x in os.listdir(data_stock) if '.' not in x]

prices = os.path.join(base, 'prices')

files = [x for x in os.listdir(prices) if not x.startswith('WIG')]
cols = pd.read_csv(os.path.join(prices, 'WIG.csv')).set_index('<DTYYYYMMDD>').columns

z = [np.repeat([x for x in range(len(cols))], len(files)+1), np.tile([x for x in range(len(cols))], len(files)+1 )]

def cnan(x):
    return np.sum(np.isnan(x))

def edgify(bins):
    return bins[:-1] + bins[:-1]/2 - bins[1:]/2

def bins_with_outsiders(data, outsiders=[-1,1], amount=51):
    mini = np.amin(data)
    maxi = np.amax(data)
    
    data[data<outsiders[0]] = outsiders[0]
    data[data>outsiders[1]] = outsiders[1]
    
    return np.linspace(outsiders[0], outsiders[1], amount)


def cinf(x):
    return np.sum(np.isinf(x) | np.isneginf(x))

def rinf(x):
    x[np.isinf(x) | np.isneginf(x)] = 0
    return x

def density_plot(data, bins):
    v, edges = np.histogram(data, bins=bins)
    v = v
    x = edges[:-1] + edges[:-1]/2 - edges[1:]/2
    return v, x

def bins_half(data):
    mini = np.round(np.amin(data), 0)-.5
    maxi = np.round(np.amax(data), 0)+1
    
    return np.linspace(mini, maxi, min( (maxi-mini)/.5, 100 ))

def bins_amount(data, amount=51):
    mini = np.round(np.amin(data), 0)-.5
    maxi = np.round(np.amax(data), 0)+1
    
    return np.linspace(mini, maxi, amount)

def bins_outliers(data, perc=[2,98], amount=51):
    
    p, q = np.percentile(data, q=perc)
    data[data<p] = p
    data[data>q] = q
    
    return np.linspace(p, q, num=amount, endpoint=True)

def plots(indicator, results, title, bins=bins_half, loc='upper right', common_denom=True, figsize=(10,8), **kwargs):
    
    where_ups = np.where(results==1)
    where_downs = np.where(results==0)
    
    bins = bins(indicator, **kwargs)
    
    A = density_plot(indicator[where_ups], bins)
    B = density_plot(indicator[where_downs], bins)
    if common_denom:
        denom = np.sum(A[0]+B[0])
        plt.plot(A[1], A[0]/denom, c='red', label='up')
        plt.plot(B[1], B[0]/denom, c='blue', label='down')
    else:
        plt.plot(A[1], A[0]/np.sum(A[0]), c='red', label='up')
        plt.plot(B[1], B[0]/np.sum(B[0]), c='blue', label='down')
        
    plt.title(title)
    return A, B


class ROC:
    def __init__(self, i, jump):
        self.i = i
        self.jump = jump
        self.samples, self.results, self.test, self.wig, self.vol, self.dates_names = self.summon(i)
        print(directories[i])
        self.results = (self.results > 0).astype(int)
    
    def summon(self,i):
        path_new = os.path.join(os.path.dirname(prices), 'simple neural network/' + directories[i])

        with open(os.path.join(path_new, 'samples.txt'), 'r') as f:
            samples = np.array(json.load(f))

        with open(os.path.join(path_new, 'results.txt'), 'r') as f:
            results = np.array(json.load(f))

        with open(os.path.join(path_new, 'test_indices.txt'), 'r') as f:
            test = json.load(f)

        with open(os.path.join(path_new, 'wig.txt'), 'r') as f:
            wig = np.array(json.load(f))

        with open(os.path.join(path_new, 'vol.txt'), 'r') as f:
            vol = np.array(json.load(f))

        with open(os.path.join(path_new, 'dates_names.txt'), 'r') as f:
            dates_names = np.array(json.load(f))

        return samples, results, test, wig, vol, dates_names
    



def transform_RSI(i, samples, wig, results):
    path_new = os.path.join(os.path.dirname(prices), 'simple neural network/' + directories[i])


    rel_samples = (samples[:, 1:] - samples[:, :-1])/samples[:, :-1]
    rel_wig = (wig[:, 1:] - wig[:, :-1])/wig[:, :-1]

    sum_up = np.sum( rel_samples > 0, axis=1)

    sum_down = np.sum( rel_samples <= 0, axis=1)

    ups = np.sum( rel_samples * (rel_samples > 0), axis = 1)/sum_up
    downs = -np.sum( rel_samples * (rel_samples <= 0), axis=1)/sum_down

    RS = ups/downs

    RSI = 100*(1 - 1/(1+RS))

    results = np.array(results)

    results_ = (results > 0).astype(int)

    where_ok = np.where(~np.isnan(RSI))

    rsi_ok = RSI[where_ok]

    return RSI, results_, results, where_ok


def transform_to_sharpe(samples, wig, results, dates_names):

    rel_samples = (samples[:, 1:] - samples[:, :-1])/samples[:, :-1]
    rel_wig = (wig[:, 1:] - wig[:, :-1])/wig[:, :-1] 
    
    sW = np.std(rel_wig, axis=1)[None].T
    
    mS = np.mean(rel_samples, axis=1)[None].T
    mW = np.mean(rel_wig, axis=1)[None].T
    
    beta = np.mean((rel_samples-mS)*(rel_wig-mW), axis=1)[None].T/sW
    
    return np.hstack([(mS - mW)/sW, (mS-mW)/beta, results[None].T])


def wigify(wig, dates_names, alpha=.8):
    
    rel_wig = (wig[:, 1:] - wig[:, :-1])/wig[:, :-1]
    
    where = np.unique(dates_names[:, 1], return_index=True)[1]
    
    z = np.hstack( [rel_wig[where, :], dates_names[where, 1][None].T ])
    del where
    
    z = z[np.argsort(z[:, -1]), :]
    
    dicts_loc = dict(zip(z[:, -1], np.arange(z.shape[0])))
    
    means = np.mean(z[:, :-1].astype(float), axis=1)
    p = np.zeros(z.shape[0])
    
    for i in np.arange(z.shape[0]):
        
        p[i] = np.average(means[:i+1], weights=np.hstack([np.cumprod(alpha*np.ones(i)), 1] ))
        
    return p, dicts_loc

def transform_to_sharpe_EMA(samples, wig, results, dates_names, alpha=.8):
    
    rel_samples = (samples[:, 1:] - samples[:, :-1])/samples[:, :-1]
    
    wig_ema, wig_dict = wigify(wig, dates_names, alpha)
    
    ind = np.argsort(dates_names[:, 0], kind='mergesort') #need stable algorithm without exceptions
    
    samples, results, dates_names = samples[ind, :], results[ind], dates_names[ind, :]
    
    comp_uniq, comp_count = np.unique(dates_names[ind, 0], return_counts=True)
    
    offset = 0
    
    overall_sharpe = np.zeros(samples.shape[0])
    
    for i in np.arange(comp_uniq.shape[0]):
        cc = comp_count[i]
        sets = set(wig_dict.keys()) & set(dates_names[offset:offset+cc, 1])
        current_dates = [wig_dict[x] for x in sets]
        
        if len(current_dates) != cc:
            was_there = set()
            indices = np.unique(dates_names[offset:offset+cc, 1], return_index=True)[1] + offset
        else:
            indices = np.arange(offset, offset+cc)
        
        L = len(indices)
        means = np.mean(rel_samples[indices, :], axis=1)
        stds = np.std(rel_samples[indices, :], axis=1)
        comp_ema = np.zeros(L)
        comp_ema_of_std = np.zeros(L)
        
        for j in np.arange(L):
            w = np.hstack([np.cumprod(alpha*np.ones(j))[::-1], 1] )
            comp_ema[j] = np.average(means[:j+1], weights = w)
            comp_ema_of_std[j] = np.average(stds[:j+1], weights = w)
            
        try:
            overall_sharpe[indices] = (comp_ema - wig_ema[current_dates].astype(float))/comp_ema_of_std
        except:
            print(comp_uniq[i], cc, comp_ema.shape, wig_ema[current_dates].shape, dates_names[offset+cc-1], dates_names[offset])
            return
        
        offset += cc
        
    return np.vstack([overall_sharpe, results]).T
        

def transform_average(samples, vol, results, alpha=0.9):
    
    zeros = np.where(np.sum(vol[:, 1:], axis=1)==0)
    vol[zeros, 1:] = np.random.randn(zeros[0].shape[0], vol.shape[1]-1)/1000
    rel_samples = (samples[:, 1:] - samples[:, :-1])/samples[:, :-1]
    rel_vol = (vol[:, 1:] - vol[:, :-1])/vol[:, :-1] 
    
    
    rel_vol[np.where(np.sum(rel_vol, axis=1)==0)[0], :] += np.random.randn()/100
    
    averages = np.average(rel_samples, weights=rel_vol, axis=1)[None].T
    averages_vol = np.average(rel_samples, weights=vol[:, 1:], axis=1)[None].T
    simple = (np.mean(samples, axis=1)/samples[:, -1])[None].T
    exp_avg = np.average(rel_samples, weights= np.hstack([1, alpha*np.ones( rel_samples.shape[1]-1 )]), axis=1)[None].T
    
    
    return np.hstack([averages, averages_vol, simple, exp_avg, results[None].T])
