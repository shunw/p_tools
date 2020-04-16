import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle

def line_format(label):
    """
    Convert time label to the format of pandas line plot
    """
    month = label.month_name()[:3]
    if month == 'Jan':
        month += f'\n{label.year}'
    return month

if __name__ == '__main__': 
    # df = pd.read_csv('covid_19.csv')
    # confirmed = df.loc[df['Case_Type'] == 'Confirmed', ['Cases', 'Date']]
    
    # confirmed = confirmed.groupby(['Date']).sum().reset_index()
    
    # confirmed['Date'] = confirmed['Date'].apply(pd.to_datetime, format = '%m/%d/%Y')
    
    # confirmed.set_index('Date', inplace = True)
    # # print (confirmed.head())
    # # print (confirmed.dtypes)
    # ax = confirmed.plot(kind = 'bar', y = 'Cases')
    # ax.set_xticklabels(map(lambda x: line_format(x), confirmed.index))
    # plt.show()
    

    f = open('train-labels-idx1-ubyte', encoding = 'utf-8')
    for line in f: 
        print (line)
        break