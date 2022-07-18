
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
from pandas.errors import ParserError
import argparse


class MonteCarlo:
    # Set the dates for data and choose stocks
    
    def __init__(self, file_name, time_frame, num_guess, output):

        self.file_name = file_name
        self.time_frame = time_frame
        #number of portfolios (/number of guesses)
        self.num_guess = num_guess
        # self.output = output
        self.end = datetime.today().date()

        if self.time_frame[-1].upper() == 'Y':
            self.start = self.end - timedelta(days = int(self.time_frame[:-1])*365)
        else:
            self.start = self.end - timedelta(days = int(self.time_frame[:-1])*30)

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

    def run(self):        
        # By randomly guessing the weights

        input = self._read_data()
        closing = []

        for stock in input.columns:
            closing.append(web.DataReader(stock.upper(),'yahoo', self.start, self.end)['Adj Close'])

        df = pd.concat(closing,axis=1)
        df.columns = input.columns

        log_ret = np.log(df/df.shift(1))
        all_weight = np.zeros((self.num_guess,len(df.columns)))
        ret_arr = np.zeros(self.num_guess)
        vol_arr = np.zeros(self.num_guess)
        SR_arr = np.zeros(self.num_guess)

        for ind in range(int(self.num_guess)):

            #random weights
            weight = np.array(np.random.random(len(input.columns)))
            #rebalancing
            weight = weight/np.sum(weight)
            #save the weights
            all_weight[ind,:] = weight
            # expected return
            ret_arr[ind] = np.sum((log_ret.mean()*weight)*252)
            # expected volatility calculated with dot product
            vol_arr[ind] = np.sqrt(np.dot(weight.T,np.dot(log_ret.cov()*252,weight)))
            # expected sharpe
            SR_arr[ind] = ret_arr[ind]/vol_arr[ind]
            
        SR_arr.max()
        #get the index location of the max sharpe
        SR_arr.argmax()
        opt_weight = all_weight[SR_arr.argmax()]

        txt = "Optimal weights: "
        for n in range(len(input.columns)):
            txt += f"{input.columns[n]}: {str(round(opt_weight[n]*100))}% "

        return self.plot(vol_arr,ret_arr,SR_arr,txt)

    def plot(self, x, y, z, title):
    
        #Figure
        plt.figure(figsize=(16,12))
        plt.scatter(x,y,c=x,cmap='plasma')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.title(title)

        max_sr_ret = y[z.argmax()]
        max_sr_vol = x[z.argmax()]

        # marking the best Sharpe
        plt.scatter(max_sr_vol, max_sr_ret, c='red', s=100, edgecolors='black')
        print(title)

        if args.output != None:
            print(f"Saving figure into file {args.output}.")
            return plt.savefig(str(args.output))
        else:
            return plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="Provide the input filename.")
    parser.add_argument("time_frame", type=str,
                        help="Provide the time frame of the analysis. Either in years (1Y) or in months (6M).")

    parser.add_argument("--num_guess", help="Number of guesses for the optimal porfolio. The higher the number the more reliable the result. Defaults to 5000.",
                        type=int, default=5000)

    parser.add_argument("--output", help="Provide a output file name with extension. Extension defaults to .png") 

    args = parser.parse_args()

    fw = MonteCarlo(args.file_name, args.time_frame, args.num_guess, args.output)
    fw.run()