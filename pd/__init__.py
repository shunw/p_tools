###############
# @author sw
# @date 2018-09-28
#######################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize


def scatter_plot_2D(X, y):
    '''
    do scatter plot (only 2D) and separate the points with different color/ marker/ legend
    input: X, y (both are dataframe; y define the positive and negative)
    output: plot
    '''
    
    pos = y==1
    neg = y==0
    
    ad = plt.scatter(X.loc[pos, X.columns[0]], X.loc[pos, X.columns[1]], marker = '^')
    not_ad = plt.scatter(X.loc[neg, X.columns[0]], X.loc[neg, X.columns[1]], marker = 'o')
    
    plt.legend((ad, not_ad), ('admitted', 'not admitted'), loc = 'lower left')
    

    # # # % Labels and Legend
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    plt.show()