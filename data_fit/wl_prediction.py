from wl_data import Work_load_Data

import datetime
from functools import partial
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

class CombineAttributes(BaseEstimator, TransformerMixin): 
    '''
    add this to add/ minus col easier & more standard
    '''
    def __init__(self, move_p_name = True, move_ttype = True, est_target = 'test_hs'):
        '''
        input: move_* is to move two columns; est_target is which target col to estimate ['tesths_per_month', 'test_hs', 'month_qty']
        ''' 
        self.move_p_name = move_p_name
        self.move_ttype = move_ttype
        self.est_target = est_target
        self.potential_target = ['test_hs', 'month_qty', 'tesths_per_month']
        self.potential_target.remove(est_target)

    def fit(self, X, y = None): 
        return self
    
    def transform(self, X, y = None): 
        if self.move_p_name: 
            X.drop(['p_name'], axis = 1, inplace = True)
        if self.move_ttype: 
            X.drop(['t_type'], axis = 1, inplace = True)
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


if __name__ == '__main__': 
    data_m = Work_load_Data()
    no_time = data_m.no_time_data()
    w_time = data_m.with_time_data()
    w_pro_time = data_m.w_begin_time()
    
    # remove unreasonable data
    no_time.drop(no_time[(no_time['phase'] == 'MP') & (no_time['p_name'] == 'Lark')].index, inplace = True)
    no_time.drop(no_time[(no_time['phase'] == 'MP') & (no_time['p_name'] == 'Stella')].index, inplace = True)
    
    # print (no_time.loc[(no_time['phase'] == 'MP') & (no_time['c_type'] == 'm') & (no_time['p_type'] == 'aio'), 'test_hs'].mean())
    
    # prepare the test dummy data
    phase = ['DE', 'MT', 'MP']
    color = ['c', 'm']
    p_type = ['sf', 'aio']
    cols_ord = ['phase', 'c_type', 'p_type']
    test_data_ls = [c for c in itertools.product(phase, color, p_type)]
    test_df = pd.DataFrame(columns = cols_ord, data = test_data_ls)
    test_df_dum = pd.get_dummies(test_df)
    
    # prepare the training data
    y_col = 'month_qty' # ['test_hs', 'month_qty', 'tesths_per_month']
    col_no_y = list(no_time.columns)
    col_no_y.remove(y_col)
    
    X = no_time[col_no_y]
    data_pre_transfer = CombineAttributes(est_target = y_col)
    data_pre_transfer.fit_transform(X)
    
    X = pd.get_dummies(X)
    y = no_time[[y_col]]
    y_2 = no_time[['tesths_per_month']]

    # ==================== module and prediction ====================
    # linear = LinearRegression()
    # linear.fit(X, y)
    # y_pred_l = linear.predict(test_df_dum)
    
    
    tree = DecisionTreeRegressor()
    tree.fit(X, y)
    y_pred_t = tree.predict(test_df_dum)
    # test_df['lr_pred'] = y_pred_l
    test_df['month_qty'] = y_pred_t

    tree_2 = DecisionTreeRegressor()
    tree_2.fit(X, y_2)
    y_pred_t = tree_2.predict(test_df_dum)
    test_df['hs_per_month'] = y_pred_t
    # print (test_df)
    # ==================== combine actual pro w/ predition ====================
    
    pre_w_project = w_pro_time.merge(test_df, left_on = ['phase', 'c_type', 'p_type'], right_on = ['phase', 'c_type', 'p_type'], )
    
    pre_w_project = pre_w_project.loc[pre_w_project.index.repeat(pre_w_project['month_qty'])]
    # pre_w_project['month_incr'] = [1] * pre_w_project.shape[0]
    pre_w_project['for_month_incr'] = pre_w_project['p_name'] + pre_w_project['phase']
    pre_w_project['month_incr']= pre_w_project.groupby('for_month_incr').cumcount()
    
    pre_w_project['new_date'] = pre_w_project.apply(_add_month, axis = 1)
    
    need_col = ['new_date', 'hs_per_month']
    w_time_pred = pre_w_project[need_col]
    
    w_time_new = w_time_pred.groupby('new_date').sum().reset_index()
    # print (w_time.head())
    # print (pre_w_project.head())

    # # ================== HIST PLOT ==================
    # # plt.hist(y_pred, bins = 50, alpha = .5, label = 'pred')
    # # plt.hist(y.to_numpy(), bins = 50, alpha = .5, label = 'act')
    # # plt.legend(loc = 'upper right')
    # # plt.show()

    # ================== BAR PLOT TIME SERIES==================
    # ax.bar(pd.to_datetime(w_time['date'], format='%Y-%m-%d'), w_time['test_hs'])
    # ax.xaxis_date()
    w_time.set_index('date', inplace = True)
    
    w_time_new.set_index('new_date', inplace = True)
    
    # w_time.plot(kind = 'bar', y = 'test_hs')
    # ax = w_time.plot.bar(x = 'date', y = 'test_hs')
    # ax.xaxis.set_major_formatter('%Y-%m-%d')
    # Note that we specify rot here

    ax = w_time.plot(kind='bar', y = 'test_hs', figsize=(20, 9), color='#2ecc71', rot=0, alpha = .5)
    ax = w_time_new.plot(kind='bar', y = 'hs_per_month', figsize=(20, 9), color='red', rot=0, ax = ax, alpha = .5)
    ax.set_xticklabels(map(lambda x: line_format(x), w_time.index))

    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
        tick.set_fontsize(6)
    plt.savefig('work_load.png')
    # plt.show()

    
    