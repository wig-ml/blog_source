from datetime import datetime
from bisect import *
import pandas as pd
import numpy as np
import os

class SemiMarkovChain:
    def __init__(self, shift, length, reach):
        self.shift = shift
        self.length = length
        self.reach = reach
        self.end_times, self.price_dict = {}, {}
        self.path_wig = os.path.dirname(os.getcwd())
        self.path_prices = os.path.join(self.path_wig, 'prices')
        self.path_diffs = os.path.join(os.path.join(self.path_wig, 'markov chains'), 'diffs')
        self.companies = os.listdir(self.path_diffs)

    def reset_dict(self):
        self.price_dict = {}

    def build(self, start_date='19900101', end_date='20151230'):
        scrape = datetime.strptime
        price_dict, vol_dict = {}, {}
        length = self.length
        A = np.vstack([np.repeat([0,1]*np.power(2,k), [np.power(2,length-1-k), 
                                                   np.power(2,length-1-k)]*np.power(2,k)) for k in range(self.length)])

        end_times = {}

        for i in range(A.shape[1]):
            c = ''.join(list(map(str,A[:,i])))
            self.price_dict[c] = [0,0]

        for comp in self.companies:
            df = pd.read_csv(os.path.join(self.path_diffs,comp)).set_index('<DTYYYYMMDD>')
            if 'price_' + str(self.shift) not in df.columns:
                continue
            
            dates = list(df.index)
            start = max(self.shift, bisect(dates,int(start_date)))
            
            end = max(self.shift, bisect(dates,int(end_date)))
            
            self.end_times[comp] = end
            p = df['price_' + str(self.shift)].as_matrix()[start:end]
            values = df['orig_prices'].as_matrix()[start:end]
            for run in range(self.shift):
                things = p[run::self.shift]
                for slice_step in range(len(things) - self.length - self.reach):
                    sl = things[slice_step:slice_step+self.length]
                    x = values[(slice_step+self.length)*self.shift]
                    y = values[(slice_step+self.length)*self.shift+self.reach]
                    result = int((y-x)/x > 0)
                    self.price_dict[ ''.join(list(map(str,sl))) ][result] += 1
                    
    def test_matrix(self):
        scores = [0,0]
        for comp in self.companies:
            df = pd.read_csv(os.path.join(self.path_diffs,comp)).set_index('<DTYYYYMMDD>')
            if 'price_' + str(self.shift) not in df.columns:
                continue

            end = self.end_times[comp]
            p = df['price_' + str(self.shift)].as_matrix()[end:]
            values = df['orig_prices'].as_matrix()[end:]
            for run in range(self.shift):
                things = p[run::self.shift]
                for slice_step in range(len(things)- self.length - self.reach):
                    sl = things[slice_step:slice_step+self.length]
                    if -1 in sl:
                        print(comp)
                    arg = ''.join(list(map(str,sl)))
                    x = values[(slice_step+self.length)*self.shift]
                    y = values[(slice_step+self.length)*self.shift+self.reach]
                    true_res = int((y-x)/x > 0)
                    pred_res = np.argmax( self.price_dict[arg] )
                    scores[int(true_res==pred_res)] += 1

        return scores[1]/sum(scores)

