import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ggplot import *
import scipy.stats
import json, os, operator, heapq
from bisect import *
from sklearn.metrics import roc_curve, accuracy_score, recall_score, precision_score, auc
from sklearn.linear_model import LogisticRegression as LR
%matplotlib inline

def density_plot(data, bins):
    v, edges = np.histogram(data, bins=bins)
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
    
#     plt.figure(figsize=figsize)
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
#     plt.legend(loc=loc)
#     plt.show()


def bayesify(samples, results, bins, wilson=True):
    
    p = np.sum(results==1)/results.shape[0]
    vals_pos = np.histogram(samples[np.where(results==1)], bins=bins)[0]
    vals_neg = np.histogram(samples[np.where(results==0)], bins=bins)[0]
    
    probs_arr = np.zeros((vals_pos.shape[0], 2)).T
    z = 1.96
    n = vals_pos + vals_neg
    
    q = vals_pos/n
    pos = q*p
    neg = (1-q)*(1-p)
    
    probs = pos/(pos+neg)
    
    if wilson:
        add_sub = z*np.sqrt(1/vals_pos*(probs*(1-probs)) + 1/(4*vals_pos*vals_pos)*z*z)

        probs_arr[0, :] = np.minimum( 1/(1+1/vals_pos*z)*(probs+1/(2*n)*z*z + add_sub), 1)
        probs_arr[1, :] = 1/(1+1/vals_pos*z)*(probs+1/(2*n)*z*z - add_sub)
        return probs_arr
    else:
        return probs


def frequentify(samples, results, bins, wilson=True):
    
    p = np.sum(results==1)/results.shape[0]
    
    vals_pos = np.histogram(samples[np.where(results==1)], bins=bins)[0]
    vals_neg = np.histogram(samples[np.where(results==0)], bins=bins)[0]
    
    probs_arr = np.zeros((vals_pos.shape[0], 2)).T
    z = 1.96
    n = vals_pos + vals_neg
    q = vals_pos/n
    
    if wilson:
        add_sub = z*np.sqrt(1/vals_pos*(q*(1-q)) + 1/(4*vals_pos*vals_pos)*z*z)

        probs_arr[0, :] = np.minimum( 1/(1+1/vals_pos*z)*(q+1/(2*n)*z*z + add_sub), 1)
        probs_arr[1, :] = 1/(1+1/vals_pos*z)*(q+1/(2*n)*z*z - add_sub)
        return probs_arr
    else:
        return q


 
def transform_to_RS(roc, indices, col):
    samples, results, dates_names, jump = roc.samples[indices], roc.results[indices], roc.dates_names[indices], roc.jump
    placeholder = np.array( list(map( lambda x: (samples[:, -1]-samples[:, -1-x])/samples[:,-1-x], jump)) ).T
    #print(samples.shape, placeholder.shape, dates_names[:, 0][None].T.shape, results[None].T.shape)
    z = np.hstack([placeholder, dates_names[:, 1][None].T, results[None].T ]).astype(float)
    r = z[np.argsort(z[:, -2]).astype(int), :]
    dates_uniq, dates_count = np.unique(r[:,-2], return_counts=True)

    offset = 0

    for i in np.arange(dates_uniq.shape[0]):

        maxi = np.amax(r[offset:offset+dates_count[i], col])
        if maxi == 0:
    #         print(dates_count[i])
            r[offset:offset+dates_count[i], col] = 0

        elif maxi < 0:
    #         print(maxi, dates_count[i])
            r[offset:offset+dates_count[i], col] = -maxi/r[offset:offset+dates_count[i], col]

        else:
            r[offset:offset+dates_count[i], col] /= maxi


        offset+=dates_count[i]

    r[r<-1] = -1
    print(r[:, col].shape, results.shape)
    return np.vstack([r[:, col], results]).T


def transform_RSI(roc, indices, dropnan=False):
    #path_new = os.path.join(os.path.dirname(prices), 'simple neural network/' + directories[i])

    samples, wig, results = roc.samples[indices], roc.wig[indices], roc.results[indices]
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
    if dropnan:
        where_ok = np.where(~np.isnan(RSI))

        RSI = RSI[where_ok]
        results = results[where_ok]
    else:
        np.put(RSI, np.where(np.isnan(RSI)), np.nanmean(RSI))
        
    return np.vstack([RSI, results]).T


def transform_to_sharpe(roc, indices):
    samples, wig, results, dates_names = roc.samples[indices], roc.wig[indices], roc.results[indices], roc.dates_names[indices]
    rel_samples = (samples[:, 1:] - samples[:, :-1])/samples[:, :-1]
    rel_wig = (wig[:, 1:] - wig[:, :-1])/wig[:, :-1] 
    
    sW = np.std(rel_wig, axis=1)[None].T
    
    mS = np.mean(rel_samples, axis=1)[None].T
    mW = np.mean(rel_wig, axis=1)[None].T
    
    beta = np.mean((rel_samples-mS)*(rel_wig-mW), axis=1)[None].T/(sW*sW)
    
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

def transform_to_sharpe_EMA(roc, indices, alpha=.8):
    samples, wig, results, dates_names = roc.samples[indices], roc.wig[indices], roc.results[indices], roc.dates_names[indices]
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

        overall_sharpe[indices] = (comp_ema - wig_ema[current_dates].astype(float))/comp_ema_of_std
        
        offset += cc
        
    return np.vstack([overall_sharpe, results]).T


def transform_average(roc, indices, alpha=.8):
    samples, vol, results = roc.samples[indices], roc.vol[indices], roc.results[indices]
    zeros = np.where(np.sum(vol[:, 1:], axis=1)==0)
    vol[zeros, 1:] = np.random.randn(zeros[0].shape[0], vol.shape[1]-1)/1000
    rel_samples = (samples[:, 1:] - samples[:, :-1])/samples[:, :-1]
    rel_vol = (vol[:, 1:] - vol[:, :-1])/vol[:, :-1] 
    
    
    rel_vol = rnan(rel_vol)
    rel_vol[np.where(np.sum(rel_vol, axis=1)==0)[0], :] += np.random.randn()/100
    
    averages = np.average(rel_samples, weights=rel_vol, axis=1)[None].T
    averages_vol = np.average(rel_samples, weights=vol[:, 1:], axis=1)[None].T
    simple = (np.mean(samples, axis=1)/samples[:, -1])[None].T
    exp_avg = np.average(rel_samples, weights= np.hstack([1, alpha*np.ones( rel_samples.shape[1]-1 )]), axis=1)[None].T
    
    
    return np.hstack([averages, averages_vol, simple, exp_avg, results[None].T])


def learn_probs(samples, results, bins, mode, proper_edges=True, **kwargs):
    
    bins = bins(samples, **kwargs)
    
    b = bayesify(samples, results, bins, wilson=True)
    f = frequentify(samples, results, bins, wilson=True)
    e = np.arange(b.shape[1])
    if proper_edges:
        e = edgify(bins)
    bayes_wilson = dict(zip( e, b[mode, :] ))
    freq_wilson = dict(zip( e, f[mode, :] ))
    
    b = bayesify(samples, results, bins, wilson=False)
    f = frequentify(samples, results, bins, wilson=False)
    
    bayes = dict(zip(e, b))
    freq = dict(zip(e, f))
    
    return {'Bayes wilson': bayes_wilson, 'Frequent wilson': freq_wilson, 
            'Ordinary Bayes': bayes, "Ordinary frequentist": freq}, bins

def test_probs(samples, bins, prob_dict):
    probs = np.zeros((samples.shape[0], 4))
    L = len(bins)
    k = ['Bayes wilson', 'Frequent wilson', 'Ordinary Bayes', 'Ordinary frequentist']
    exc = 0
    for j, ind in enumerate(samples):
        
#         print(bins)
#         print(bisect_left(bins,ind), L-1)
        try:
            z = [prob_dict[x][min(bisect_left(bins,ind), L-1)] for x in k]
        except:
            exc += 1
            z = [prob_dict[x][min(bisect_left(bins,ind), L-2)] for x in k]
        
        probs[j, :] = np.array(z)
                 
#     print(exc)
    return probs

def roc_rpa_curves(probs, res_tests, computation_threshold=100, plotting=False):
    if plotting:
        plt.figure(figsize=(15,12))
    C = probs.shape[1]
#     print(C)
    all_thresholds = []
    aucs = []
    for k in range(C):
#         print(k)
        tpr, fpr, thresholds = roc_curve(res_tests, probs[:, k])
        aucs.append( auc(tpr, fpr) )
        if plotting:
            plt.subplot(C,2,2*k+1)
            plt.plot(tpr, fpr, c='black')
            plt.plot(tpr, tpr, c='orange', alpha=.5)

        
        points = np.zeros((thresholds.shape[0], 3))
        L = len(thresholds)
        j = 1
        if L > computation_threshold:
            j = L//computation_threshold
            thresholds = thresholds[::j]
            
        for _, thr in enumerate(thresholds):
            r = recall_score(res_tests, probs[:, k] > thr)
            p = precision_score(res_tests, probs[:, k] > thr)
            a = accuracy_score(res_tests, probs[:, k] > thr)
#             print(_)

            points[_, :] = np.array([r,p,a])

#         print(points[::j, 2])
        if plotting:
            plt.subplot(C,2,2*k+2)
            plt.plot(thresholds, points[::j, 0], c='black', label='recall')
            plt.plot(thresholds, points[::j, 1], c='blue', label='precision')
            plt.plot(thresholds, points[::j, 2], c='red', label='accuracy')
            plt.axhline(0.5, 0, 1, c='green')
            plt.xlim(0,1)
        all_thresholds.append(thresholds)
        
    if plotting:
        plt.show()
    return all_thresholds, aucs


def testing_pipeline(roc, transformation, avg_col=0, bins=bins_outliers, mode=0, roc_jump=0, plotting=True,dropnan=False,alpha=.8,**kwargs):
    tests = roc.test
    nottests = not_tests(tests, roc.samples.shape[0])
    transfs = {'RSI':[transform_RSI, {'dropnan':dropnan}],
               'RS':[transform_to_RS, {'col':roc_jump}],
               'sharpe':[transform_to_sharpe, {}], 
               'sharpe_EMA':[transform_to_sharpe_EMA, {'alpha':alpha}], 
               'avgs':[transform_average, {'alpha':alpha}]}
               
    wh = transfs[transformation]
    
    indicator = wh[0](roc, nottests, **wh[1])
    avg_col = max(0, min(indicator.shape[1]-1, avg_col))
    indicator = indicator[:, [avg_col, -1]]
#     print(cnan(indicator), np.amin(indicator[:, 0]), np.amax(indicator[:, 0]))
    lp, bins = learn_probs(indicator[:, 0], indicator[:, 1], bins, mode, proper_edges=False, **kwargs)
#     print('learned probabilities')
    indicator_test = wh[0](roc, tests, **wh[1])[:, [avg_col, -1]]
    probs = test_probs(indicator_test[:, 0], bins, lp)
#     print('created probability array')
    thr, aucs = roc_rpa_curves(probs, indicator_test[:,1], plotting=plotting)
    ret = [thr, probs, aucs]
    return ret


def corrs(roc, bins=bins_outliers, alpha=.8, dropnans=True, **kwargs):
    tests = roc.test
    nottests = not_tests(tests, roc.samples.shape[0])
    axes = []
    
    transfs = {'RSI':[transform_RSI, {'dropnan':False}, 1], 
               'sharpe':[transform_to_sharpe, {}, 2], 
               'sharpe_EMA':[transform_to_sharpe_EMA, {'alpha':alpha}, 1], 
               'avgs':[transform_average, {'alpha':alpha}, 4]}
    
    probs = np.zeros((len(nottests), 1+2+1+4))
    i = 0
    for w,tr in transfs.items():
        func, kw, cols = tr
        indicator = func(roc, nottests, **kw)
        print(w)
        for col in range(cols):
            axes.append(w + str(col))
            vals = indicator[:, col]
#             print(cnan(vals))
            probs[:, i] = vals
            i += 1
            
    
    M = np.corrcoef(probs, rowvar=0)
    if dropnans:
        correct = [x for x in range(M.shape[0]) if cnan(M[:, x]) != M.shape[0]]
        return M[correct, :][:, correct], np.asarray(axes)[correct]
    
    return matrix, axes
            
        

def aucs(roc):
    transfs = {'RSI':1, 
               'sharpe':2, 
               'sharpe_EMA':1, 
               'avgs':4}
    
    C = np.zeros((1+2+1+4, 4)) #4 columns for different probability types
    j = 0
    for transformation, columns in transfs.items():
        print(transformation)
        for col in range(columns):
            auc = testing_pipeline(roc, transformation=transformation, plotting=False, avg_col=col)[2]
            C[j, :] = auc
            j+=1
            
    return C
            
