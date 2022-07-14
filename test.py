

import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import scipy


time_frame = 1
END = datetime.today().date()
vuosi = END.year-time_frame
START = pd.to_datetime(f"{str(vuosi)}-{END.month}-{END.day}").date()
x = web.DataReader('AAPL', 'yahoo', START, END)['Adj Close']
print(x)

print(None)