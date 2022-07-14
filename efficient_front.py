

import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize


class Plot():
    # Set the dates for data and choose stocks


    time_frame = 1
    END = datetime.today().date()
    vuosi = END.year-time_frame
    START = pd.to_datetime(f"{str(vuosi)}-{END.month}-{END.day}").date()
    LST = ['META','AMZN','GOOG','MSFT']
    
    def __init__(self, rounds):
        # self.time_frame = time_frame

        #number of portfolios (/number of guesses)
        self.rounds = rounds

    def guessing(self):        
        # By randomly guessing the weights

        closing  = []

        for i in range(len(self.LST)):
            closing.append(web.DataReader(self.LST[i],'yahoo', self.START, self.END)['Adj Close'])

        df = pd.concat(closing,axis=1)
        df.columns = self.LST

        log_ret = np.log(df/df.shift(1))
        all_weight = np.zeros((self.rounds,len(df.columns)))
        ret_arr = np.zeros(self.rounds)
        vol_arr = np.zeros(self.rounds)
        SR_arr = np.zeros(self.rounds)

        for ind in range(self.rounds):

            #random weights
            weight = np.array(np.random.random(len(self.LST)))
            #rebalancing
            weight = weight/np.sum(weight)
            #save the weights
            all_weight[ind,:] = weight
            # expected return
            ret_arr[ind] = np.sum ((log_ret.mean()*weight)*252)
            
            # expected volatility
            vol_arr[ind] = np.sqrt(np.dot(weight.T,np.dot(log_ret.cov()*252,weight)))
            
            # expected sharpe
            SR_arr[ind] = ret_arr[ind]/vol_arr[ind]
            
        SR_arr.max()
        #get the index location of the max sharpe
        SR_arr.argmax()

        max_sr_ret = ret_arr[SR_arr.argmax()]
        max_sr_vol = vol_arr[SR_arr.argmax()]
        opt_weight = all_weight[SR_arr.argmax()]

        txt = "Optimal weights: "
        for n in range(len(self.LST)):
            txt += f"{self.LST[n]}: {str(round(opt_weight[n]*100))}% "

        print(txt)

        #Figure
        plt.figure(figsize=(16,12))
        plt.scatter(vol_arr,ret_arr,c=SR_arr,cmap='plasma')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.title(txt)

        # marking the best Sharpe
        plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black')

        all_weight[SR_arr.argmax(),:] 

        print(None)

    def optimization(self, df):
        #With the optimization tool

        log_ret = np.log(df/df.shift(1))

        def _get_ret_vol_sr(weights):

            weights = np.array(weights)
            ret = np.sum(log_ret.mean()*weights)*252
            vol = np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252,weights)))
            sr = ret/vol
            return np.array([ret,vol,sr])

        # returns the negative sharpe ratio from the given array
        def _neg_sharpe(weights):
            return _get_ret_vol_sr(weights)[2] * -1

        # returns 0 if the sum of the weights == 1
        def _check_sum(weights):
            return np.sum(weights) - 1 

        cons = ({'type': 'eq','fun':_check_sum})

        bounds_list = []
        init_guess = []

        for n in range(len(self.LST)):    
            bounds_list.append((0,1))
            init_guess.append(1/len(self.LST))
            
        bounds = tuple(bounds_list)
        #running the minimization algoritgm
        opt_results = minimize(_neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
        
        #the resulting weights are held in x array
        _get_ret_vol_sr(opt_results.x)

        texti = ''
        for m in range(len(opt_results.x)):
            texti += self.LST[m]+": "+ str(round(opt_results.x[m]*100))+'% '
            
        print(texti)
        
        # Efficient frontier
        xrange_left = 0.17
        xrange_right = 0.34
        frontier_y = np.linspace(xrange_left,xrange_right,100)

        def _minimize_vol(weights):
            return _get_ret_vol_sr(weights)[1]

        frontier_vol = []

        for possible_return in frontier_y:
            cons = ({'type':'eq','fun': _check_sum},
                    {'type':'eq','fun': lambda w: _get_ret_vol_sr(w)[0]-possible_return})
            
            result = minimize(_minimize_vol,init_guess,method='SLSQP',bounds=bounds,constraints=cons)

            frontier_vol.append(result['fun'])
            
        #Figure
        # plt.figure(figsize=(16,12))
        # plt.scatter(vol_arr,ret_arr,c=SR_arr,cmap='plasma')
        # plt.colorbar(label='Sharpe Ratio')
        # plt.xlabel('Volatility')
        # plt.ylabel('Return')
        # plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black',label=texti)
        # plt.legend()
        plt.plot(frontier_vol,frontier_y,'g--' )

        print(None)


# if __name__ == "main":
inst = Plot(1000)
inst.guessing()

