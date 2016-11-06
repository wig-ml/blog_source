import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from bisect import bisect, bisect_left
import heapq, operator, os
from datetime import datetime, timedelta

class NewDict(dict):
    
    def add(self, another):
        c = {}
        my_keys = set(self.keys())
        his_keys = set(another.keys())
        
        for key in my_keys & his_keys:
            c[key] = self[key] + another[key]
        
        for key in my_keys - his_keys:
            c[key] = self[key]
            
        for key in his_keys - my_keys:
            c[key] = another[key]
            
        self = c
        return NewDict(self)
    
    def top(self, K, greater_then=None, lesser_then=None):
        D = dict(zip(*(zip(*heapq.nlargest(K, self.items(), key=operator.itemgetter(1))))))
        C = {}
        if greater_then:
            for key, value in D.items():
                if value > greater_then:
                    C[key] = value
        return C


class PortfolioManager:
    def __init__(self, classifiers, cash):
        self.cash = cash
        self.clfs = classifiers
        self.reach = classifiers[0].reach
        self.file_list = {}
        for f in os.listdir(files_path):
            try:
                if f.startswith('WIG'):
                    print(f)
                    continue
                self.file_list[f] = pd.read_csv(os.path.join( files_path, f )).set_index('<DTYYYYMMDD>')
                self.file_list[f] = self.file_list[f][['orig_prices'] + ['price_' + str(x) for x in [1,2,5,7,10,14]]]
            except KeyError:
                continue
        self.portfolio = {}
        print('portfolio created')
        
    def check_strategy(self, start_date):
        current_date = datetime.strptime(start_date, '%Y%m%d')
        while current_date < datetime(2016,10,28):
            fails = 0
            
            probs_clf_dict = {}
            for clf in self.clfs:
                s = clf.shift
                l = clf.length
                key = 'L' + str(l) + 'S' + str(s)
                probs_clf_dict[key] = []
                for x, file in self.file_list.items():
                    dates = list(map(str, file.index))

                    current_index = bisect_left(dates, current_date.strftime('%Y%m%d'))
                    if current_index < s*(l+1):
                        continue
                        
                    if current_date - datetime.strptime(dates[current_index-1], '%Y%m%d') > timedelta(days=self.reach-1):
                        continue

                    prices = file['orig_prices']
                    changes = file['price_' + str(clf.shift)]
                    try:
                        predict = clf.price_dict[ ''.join(list(map(str, changes[current_index:current_index-s*(l):-s])))[::-1]]
                    except:
                        fails += 1
                        continue
                    scores = res_dicts[key]['scores']
                    
                    p = predict[1]/sum(predict)
                    arg = np.argmax(predict)
                    
                    recall = {0: scores[0,0]/sum(scores[:, 0]), 1:scores[1, 1]/sum(scores[:, 1])}
                    
                    if arg:
                        up_prob = p*recall[1]/(p*recall[1]+(1-p)*(1-recall[0]))
                    else:
                        up_prob = p*recall[1]/(p*(1-recall[1])+(1-p)*recall[0])
                    
                    probs_clf_dict[key].append([x, up_prob, 1-up_prob])
                    
                probs = np.asarray(probs_clf_dict[key])
                ups = list(zip(*heapq.nlargest(5, probs[:,[0,1]], key=operator.itemgetter(1) )))
                downs = list(zip(*heapq.nlargest(5, probs[:,[0,2]], key=operator.itemgetter(1) )))
                probs_clf_dict[key] = {'ups':dict(zip(*ups)), 'downs':dict(zip(*downs))}
                #przydalby sie przedzial ufnosci dla kazdej zmiennej, ale na razie zadowolimy sie srednimi
                
            app_up, app_down = NewDict(), NewDict()
            for key in probs_clf_dict.keys():
                app_up = app_up.add({k:1 for k in probs_clf_dict[key]['ups'].keys()})
                app_down = app_down.add({k:1 for k in probs_clf_dict[key]['downs'].keys()})
                            
            app_up = app_up.top(5, greater_then=1)
            app_down = app_down.top(5, greater_then=1)
            for comp, amount in self.portfolio.items():
                if comp in app_down:
                    df = self.file_list[comp]
                    dates = list(map(str,df.index))
                    current_index = bisect_left(dates, current_date.strftime('%Y%m%d'))
                    prices = df['orig_prices']
                    self.cash += prices.iloc[current_index]*amount
            
            sum_app_up = sum([val for val in app_up.values()])
            
            for comp, app in app_up.items():
                
                df = self.file_list[comp]
                dates = list(map(str,df.index))
                ind = bisect_left(dates, current_date.strftime('%Y%m%d'))
                prices = df['orig_prices']
                cash = app/sum_app_up*self.cash
                amount = int(cash/prices.iloc[ind - int(ind==len(df))])
                self.cash -= amount*prices.iloc[ind - int(ind==len(df))]
                if comp not in self.portfolio:
                    self.portfolio[comp] = 0
                self.portfolio[comp] += amount
                                    
                    
            
            self.last_update = current_date
            print(current_date, self.cash, fails, self.check_net_worth())
            current_date  += timedelta(days=self.reach)
            
    def check_net_worth(self):
        c = 0
        c += self.cash
        for comp, amount in self.portfolio.items():
            df = self.file_list[comp]
            dates = list(map(str,df.index))
            ind = bisect_left(dates, self.last_update.strftime('%Y%m%d'))
            prices = list(df['orig_prices'])
            c += amount*prices[ind-int(ind==len(df))]
        return c
            
    
    
