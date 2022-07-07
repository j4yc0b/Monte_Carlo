

import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy


class Plot():

    # Set the dates for data and choose stocks

    END = datetime.date.today()
    START = datetime.datetime(int(str(END).split('-')[0])-2, int(str(END).split('-')[1]), int(str(END).split('-')[2]))
    LST = ['AAPL','AMZN','GOOG','MSFT']
    #,'GM','WMT','RDSB.L','NFLX','MCD','NKE','BA']
    
    def run(self):
        closing  = []
        for i in range(len(self.LST)):
            closing = web.DataReader(self.LST[i], 'yahoo', self.START, self.END)['Adj Close']
            cum_returns = closing[i].iloc[0] / closing[i]

        stocks = pd.concat(closing, cum_returns, axis=1)
        # stocks.columns = self.LST

        return print(stocks)

    def rest(self):
        
        # By randomly guessing the weights

        log_ret = np.log(stocks/stocks.shift(1))
        num_por = 5000
        all_weight = np.zeros((num_por,len(stocks.columns)))
        ret_arr = np.zeros(num_por)
        vol_arr =np.zeros(num_por)
        SR_arr = np.zeros(num_por)

        for ind in range(num_por):

            #weights
            weight = np.array(np.random.random(len(lst)))
            weight = weight/np.sum(weight)
            
            #save the weights
            all_weight[ind,:] = weight
            
            #return
            ret_arr[ind] = np.sum ((log_ret.mean()*weight)*252)
            
            #volatility
            vol_arr[ind] = np.sqrt(np.dot(weight.T,np.dot(log_ret.cov()*252,weight)))
            
            
            #sharpe
            SR_arr[ind] = ret_arr[ind]/vol_arr[ind]
            
            
        SR_arr.max()
        SR_arr.argmax()

        max_sr_ret = ret_arr[SR_arr.argmax()]
        max_sr_vol = vol_arr[SR_arr.argmax()]

        #Figure
        plt.figure(figsize=(16,12))
        plt.scatter(vol_arr,ret_arr,c=SR_arr,cmap='plasma')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')

        plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black')


        all_weight[SR_arr.argmax(),:] 


        #With the optimization tool

        def get_ret_vol_sr(weights):
            weights = np.array(weights)
            ret = np.sum(log_ret.mean()*weights)*252
            vol = np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252,weights)))
            sr = ret/vol
            return np.array([ret,vol,sr])

        from scipy.optimize import minimize

        def neg_sharpe(weights):
            return get_ret_vol_sr(weights)[2] * -1

        def check_sum(weights):
            return np.sum(weights) - 1 


        cons = ({'type': 'eq','fun':check_sum})

        bounds_list = []
        init_guess = []

        for n in range(len(lst)):    
            bounds_list.append((0,1))
            init_guess.append(1/len(lst))
            
        bounds = tuple(bounds_list)


        opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)

        get_ret_vol_sr(opt_results.x)

        texti = ''
        for m in range(len(opt_results.x)):
            texti += lst[m]+": "+ str(round(opt_results.x[m]*100))+'% '
            
        texti
        
        # Efficient frontier

        frontier_y = np.linspace(0.17,0.34,100)

        def minimize_vol(weights):
            return get_ret_vol_sr(weights)[1]

        frontier_vol = []


        for possible_return in frontier_y:
            cons = ({'type':'eq','fun': check_sum},
                    {'type':'eq','fun': lambda w: get_ret_vol_sr(w)[0]-possible_return})
            
            result = minimize(minimize_vol,init_guess,method='SLSQP',bounds=bounds,constraints=cons)

            frontier_vol.append(result['fun'])
            
        #Figure
        plt.figure(figsize=(16,12))
        plt.scatter(vol_arr,ret_arr,c=SR_arr,cmap='plasma')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black',label=texti)
        plt.legend()
        plt.plot(frontier_vol,frontier_y,'g--' )



if __name__ == "main":
    inst = Plot()
    inst.run()  

print(None)
