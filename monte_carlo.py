
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
from pandas.errors import ParserError


class Plot:
    # Set the dates for data and choose stocks
    
    def __init__(self, file_name, time_frame, rounds):

        format = '1Y, 3M' 
        self.file_name = file_name
        #number of portfolios (/number of guesses)
        self.rounds = rounds
        end = datetime.today().date()

        if time_frame[-1].upper() == 'Y':
            vuosi = end.year-int(time_frame[:-1])
            start = pd.to_datetime(f"{str(vuosi)}-{end.month}-{end.day}").date()
        else:
            pass

    def _read_data(self):

        csv_delimiter = ','
        if os.path.exists(self.file_name):
            try:
                input = pd.read_csv(self.file_name, sep=csv_delimiter)
                if input.shape[0] != 0:
                    raise Exception("Please provide input tickers in one line.")
            except ParserError:
                raise Exception("Failed to read in the csv data. Please provide the correct delimiter.")
        else:
            raise Exception(F"Given argument {self.file_name} is not a valid path.")

        return input

    def guessing(self):        
        # By randomly guessing the weights

        input = self._read_data()
        closing  = []

        for stock in input.columns:
            closing.append(web.DataReader(stock.upper(),'yahoo', self.start, self.end)['Adj Close'])

        df = pd.concat(closing,axis=1)
        df.columns = input.columns

        log_ret = np.log(df/df.shift(1))
        all_weight = np.zeros((self.rounds,len(df.columns)))
        ret_arr = np.zeros(self.rounds)
        vol_arr = np.zeros(self.rounds)
        SR_arr = np.zeros(self.rounds)

        for ind in range(self.rounds):

            #random weights
            weight = np.array(np.random.random(len(input.columns)))
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
        for n in range(len(input.columns)):
            txt += f"{input.columns[n]}: {str(round(opt_weight[n]*100))}% "

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


# if __name__ == "main":
inst = Plot("input.csv", "1y", 500)
inst.guessing()

