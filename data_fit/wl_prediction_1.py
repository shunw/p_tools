from wl_data import Work_load_Data

import datetime
from functools import partial
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

class CombineAttributes(BaseEstimator, TransformerMixin): 
    '''
    add this to add/ minus col easier & more standard
    '''
    def __init__(self, move_p_name = True, move_ttype = True, est_target = 'test_hs', w_month = False):
        '''
        input: move_* is to move two columns; est_target is which target col to estimate ['tesths_per_month', 'test_hs', 'month_qty']
        ''' 
        self.move_p_name = move_p_name
        self.move_ttype = move_ttype
        self.est_target = est_target
        self.potential_target = ['test_hs', 'month_qty', 'tesths_per_month']
        self.potential_target.remove(est_target)
        self.w_month = w_month
        

    def fit(self, X, y = None): 
        return self
    
    def transform(self, X, y = None): 
    
        if self.move_p_name: 
            X.drop(['p_name'], axis = 1, inplace = True)
        if self.move_ttype: 
            X.drop(['t_type'], axis = 1, inplace = True)
        
        if not self.w_month: 
    
            for i in self.potential_target: 
                X.drop([i], axis = 1, inplace = True)
        return X
        
def _add_month(x): 
    return x['date'] + DateOffset(months= x['month_incr'])

def line_format(label):
    """
    Convert time label to the format of pandas line plot
    """
    month = label.month_name()[:3]
    if month == 'Jan':
        month += f'\n{label.year}'
    return month

def plot_learning_curves(model, X, y): 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .2)
    train_errors, val_errors = [], []

    for m in range(1, len(X_train)): 
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    
    plt.plot(np.sqrt(train_errors), 'r-+', linewidth = 2, label = 'train')
    plt.plot(np.sqrt(val_errors), 'b-', linewidth = 3, label = 'val')
def get_dummy_data(max_month_predict = 15): 
    '''
    prepare the dummy data for the further predict
    return the data with 
        1. no month_incr, col not deal with dummy  
        2. no month_incr
        3. with month_incr, col not deal with dummy
        3. with month_incr
    '''
    # ================ prepare the test dummy data ================ 

    phase = ['DE', 'MT', 'MP']
    color = ['c', 'm']
    p_type = ['sf', 'aio']
    # cols_ord = ['phase', 'c_type', 'p_type']
    cols_ord = ['phase', 'c_type', 'p_type', 'month_incr']
    month_incr_ls = list(range(1, max_month_predict))
    
    # only with project, no each month
    # =============================================================
    
    test_data_ls = [c for c in itertools.product(phase, color, p_type)]
    test_df = pd.DataFrame(columns = cols_ord[: -1], data = test_data_ls)
    test_df_dum = pd.get_dummies(test_df)

    # with project, with each month
    # =============================================================
    test_data_ls_m = [c for c in itertools.product(phase, color, p_type, month_incr_ls)]
    test_df_m = pd.DataFrame(columns = cols_ord, data = test_data_ls_m)
    test_df_dum_m = pd.get_dummies(test_df_m)

    return test_df, test_df_dum, test_df_m, test_df_dum_m

def plot_month_compare_bar(w_time, w_time_new, save_fig = False):
    '''
    input: w_time; w_time_new are dataframe. 
        two main cols: date (for month); test_hs for each month
        cols name: date; test_hs/ new_date; hs_per_month
    overall: show bar plot, overlaped with transparency. 
    purpose: to check the how the monthly prediction performance. 
        w_time test hours totaly for each month (actual); vs the w_time_new test hours totall for each month (prediction). 
    ''' 
    w_time.set_index('date', inplace = True)
    
    w_time_new.set_index('new_date', inplace = True)

    ax = w_time.plot(kind='bar', y = 'test_hs', figsize=(20, 9), color='#2ecc71', rot=0, alpha = .5)
    ax = w_time_new.plot(kind='bar', y = 'hs_per_month', figsize=(20, 9), color='red', rot=0, ax = ax, alpha = .5)
    ax.set_xticklabels(map(lambda x: line_format(x), w_time.index))

    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
        tick.set_fontsize(6)

    if save_fig: 
        plt.savefig('work_load_default.png')
    else: 
        plt.show()

if __name__ == '__main__': 
    pass