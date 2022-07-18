This code can be used to run a Monte Carlo simulation for determining the optimal weights of a given portfolio in the provided time frame, as well as determining the Sharpe ratio of this portfolio. 
User can run the code simply from the command line and providing the required command line arguments. The first 
argument takes in a .csv file. The tickers of the portfolio needs to be provided in this file in a single line.

Required
file_name: The file path of the csv-file. 
time_frame: The time frame of the analysis. Either in years (1Y) or in months (6M).

Optional
num_guess: Number of guesses for the optimal porfolio.
output: The output file name with extension.