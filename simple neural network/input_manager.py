import numpy as np
import pandas as pd
import tensorflow as tf
import os, json, heapq, operator
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from bisect import bisect, bisect_left, bisect_right

class InputManager:
    def __init__(self, reach, shift, length):
        self.R = reach
        self.S = shift
        self.L = length
        self.file_list = {f: pd.read_csv(os.path.join(price_path, f)).set_index('<DTYYYYMMDD>') for f in os.listdir(price_path) if not f.startswith('WIG')}
        self.samples = []
        self.labels = []
        self.encoding = 'R_' + str(self.R) + 'S_' + str(self.S)
        self.fatalities = 0
        
    def extract_data(self, start_date='19900101', end_date='20140101'):
        for name, file in self.file_list.items():
            dates = list(map(str, file.index))
            start_ind = bisect(dates, start_date)
            end_ind = bisect(dates, end_date)
            final = bisect(dates, (datetime.strptime(end_date, '%Y%m%d')+timedelta(days=self.R+3)).strftime('%Y%m%d'))
            if end_ind == final:
                end_ind -= self.R
            dates = dates[start_ind:final]
            dates_ind_dict = dict(zip(dates, list(range(len(dates)))))
            if len(dates) == 0 or end_ind < 0:
                continue
            
            prices = file['<CLOSE>'].as_matrix()[start_ind:final]
            
            
            self.comp_sample = []
            self.comp_labels = []
            end_date_dt = datetime.strptime(dates[end_ind-start_ind], '%Y%m%d')
            
            
            for run in range(self.S):
                try:
                    things = prices[run:]

                    current_date = datetime.strptime(dates[run], '%Y%m%d')
                    slices = []
                    respective_dates = []
                    while current_date < end_date_dt:
                        c = current_date.strftime('%Y%m%d')
                        while current_date < end_date_dt and c not in dates_ind_dict:
                            current_date += timedelta(days=1)
                            c = current_date.strftime('%Y%m%d')

                        if current_date == end_date_dt:
                            break
                        slices.append(prices[dates_ind_dict[c]])
                        respective_dates.append(dates[dates_ind_dict[c]])
                        current_date += timedelta(days=self.S)

                    for nr in range(len(slices)-self.L-1):

                        last_date = respective_dates[nr+self.L+1]
                        check_date = datetime.strptime(last_date, '%Y%m%d')+timedelta(days=self.R)
                        while check_date < end_date_dt + timedelta(days=self.R+3) and check_date.strftime('%Y%m%d') not in dates_ind_dict:
                            check_date += timedelta(days=1)

                        if check_date.strftime('%Y%m%d') not in dates_ind_dict:
                            continue
                        self.samples.append(slices[nr:nr+self.L+1])
                        result = int((prices[dates_ind_dict[check_date.strftime('%Y%m%d')]] - prices[dates_ind_dict[last_date]]) > 0)
                        if result == 1:
                            self.labels.append([0,1])
                        elif result == 0:
                            self.labels.append([1,0])
                except:
                    print('something bad happend at', name, ' while run number', run)
                    continue
        
        self.samples = np.array(self.samples)
        
    def scale(self):
        if len(self.samples)==0:
            print('no samples to preprocess')
        sample_mean = np.mean(self.samples)
        sample_std = np.std(self.samples)
        self.samples = (self.samples-sample_mean)/sample_std
        

    def polynom(self, inplace=True, degree=2):
        if len(self.samples)==0:
            print('no samples to preprocess')
            
        pf = PF(degree=degree, interaction_only=True, include_bias=False)
        if inplace:
            self.samples = pf.fit_transform(self.samples)
            return
        return pf.fit_transform(self.samples)
        
    
    
    def rel(self, inplace=False):
        if len(self.samples) == 0:
            print('no samples to preprocess')
            
        C = (self.samples[:, -1][None].T - self.samples[:, :-1])/self.samples[:, :-1]
        if inplace:
            self.samples = C
            return
        return C
        
    def get_data(self):
        return self.samples, self.labels
                
                    
        
                
            
        

                
                    
        
                
            
        

