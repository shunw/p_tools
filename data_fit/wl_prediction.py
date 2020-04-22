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

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

if __name__ == '__main__': 
    data_m = Work_load_Data()
    no_time = data_m.no_time_data() # no time, actual test hours
    w_time = data_m.with_time_data()
    w_pro_time = data_m.w_begin_time()
    pro_w_each_m = data_m.proj_each_month()

    no_time_trd = data_m.w_begin_time_tradition() # traditional calculation
    
    # remove unreasonable data
    no_time.drop(no_time[(no_time['phase'] == 'MP') & (no_time['p_name'] == 'Lark')].index, inplace = True)
    no_time.drop(no_time[(no_time['phase'] == 'MP') & (no_time['p_name'] == 'Stella')].index, inplace = True)
    
    # print (no_time.loc[(no_time['phase'] == 'MP') & (no_time['c_type'] == 'm') & (no_time['p_type'] == 'aio'), 'test_hs'].mean())
    
    # ================ prepare the test dummy data ================ 

    phase = ['DE', 'MT', 'MP']
    color = ['c', 'm']
    p_type = ['sf', 'aio']
    # cols_ord = ['phase', 'c_type', 'p_type']
    cols_ord = ['phase', 'c_type', 'p_type', 'month_incr']
    month_incr_ls = list(range(1, pro_w_each_m['month_incr'].max() + 2))
    
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

    # ================ prepare the training data ================ 

    # only with project, no each month
    # =============================================================
    y_col = 'month_qty' # ['test_hs', 'month_qty', 'tesths_per_month']
    col_no_y = list(no_time.columns)
    col_no_y.remove(y_col)
    
    X = no_time[col_no_y]
    data_pre_transfer = CombineAttributes(est_target = y_col)
    data_pre_transfer.fit_transform(X)
    X = pd.get_dummies(X)
    y = no_time[[y_col]]

    scaler = StandardScaler()
    X_trans = scaler.fit_transform(X)
    y_2 = no_time[['tesths_per_month']]

    # with project, with each month
    # =============================================================
    y_col_m = 'test_hs'

    col_no_y_m = list(pro_w_each_m.columns)
    col_no_y_m.remove(y_col_m)
    
    Xm = pro_w_each_m[col_no_y_m]
    data_pre_transfer_m = CombineAttributes(est_target = 'test_hs', w_month = True)
    data_pre_transfer_m.fit_transform(Xm)
    Xm = pd.get_dummies(Xm)

    scaler_m = StandardScaler()
    Xm_scale = scaler_m.fit_transform(Xm)
    
    ym = pro_w_each_m[[y_col_m]]
    
    # ==================== module and prediction ====================
    linear = LinearRegression()
    # linear.fit(X, y)
    # y_pred_l = linear.predict(test_df_dum)
    
    tree = DecisionTreeRegressor()

    rf = RandomForestRegressor(n_estimators = 100)
    ef = ExtraTreesRegressor(n_estimators = 100)
    grbt = GradientBoostingRegressor(n_estimators= 100)

    svr = SVR(gamma= 'scale')

    test_alg = clone(svr)
    test_alg_2 = clone(svr)
    test_alg_m = clone(svr)
    
    # only with project, no each month
    # =============================================================
    test_df_dum_scale = scaler.transform(test_df_dum)
    test_alg.fit(X_trans, y)
    y_pred_t = test_alg.predict(test_df_dum_scale)
    # test_df['lr_pred'] = y_pred_l
    test_df['month_qty'] = y_pred_t

    test_alg_2.fit(X_trans, y_2)
    y_pred_t = test_alg_2.predict(test_df_dum_scale)
    test_df['hs_per_month'] = y_pred_t
    # print (test_df)
    
    # with project, with each month
    # =============================================================
    test_df_dum_m_scale = scaler_m.transform(test_df_dum_m)

    test_alg_m.fit(Xm_scale, ym)
    y_pred_m = test_alg_m.predict(test_df_dum_m_scale)
    test_df_m['test_hs_m_pred'] = y_pred_m
    print (test_df_m.shape)
    print (test_df_m.head())
    # # ==================== check learning curve ====================
    # plot_learning_curves(test_alg, X, y)
    # plt.show()
    
    # # ================== HIST PLOT ==================
    # # plt.hist(y_pred, bins = 50, alpha = .5, label = 'pred')
    # # plt.hist(y.to_numpy(), bins = 50, alpha = .5, label = 'act')
    # # plt.legend(loc = 'upper right')
    # # plt.show()
   
    # ================== BAR PLOT TIME SERIES==================

    # ==================== combine actual pro w/ predition ====================

    pre_w_project = w_pro_time.merge(test_df, left_on = ['phase', 'c_type', 'p_type'], right_on = ['phase', 'c_type', 'p_type'], )
    
    pre_only_project = pre_w_project.copy() # only project and its total test hours

    # with time, add the test hours / month
    # ===================================================
    pre_w_project = pre_w_project.loc[pre_w_project.index.repeat(pre_w_project['month_qty'])]
    # pre_w_project['month_incr'] = [1] * pre_w_project.shape[0]
    pre_w_project['for_month_incr'] = pre_w_project['p_name'] + pre_w_project['phase']
    pre_w_project['month_incr']= pre_w_project.groupby('for_month_incr').cumcount()
    
    pre_w_project['new_date'] = pre_w_project.apply(_add_month, axis = 1)
    
    need_col = ['new_date', 'hs_per_month']
    w_time_pred = pre_w_project[need_col]
    
    w_time_new = w_time_pred.groupby('new_date').sum().reset_index()

    # with no time, only project
    # ===================================================
    pre_only_project['total_hr_pred'] = pre_only_project['month_qty'] * pre_only_project['hs_per_month']
    
    p_need_col = ['p_name', 'phase', 'total_hr_tradition', 'total_hr_pred', 'test_hs']

    pre_only_project = pre_only_project.merge(no_time_trd, left_on = ['p_name', 'phase'], right_on = ['p_name', 'phase'])
    pre_only_project = pre_only_project.merge(no_time, left_on = ['p_name', 'phase'], right_on = ['p_name', 'phase'])[p_need_col]
    

    # ==================== cal_errors ====================

    # error for the test hours per month (avg)
    # ===================================================
    comp_test_hs = w_time.merge(w_time_new, left_on = 'date', right_on = 'new_date')
    mse_for_hs = mean_squared_error(comp_test_hs[['test_hs']], comp_test_hs[['hs_per_month']]) ** .5 # mean squared error
    
    hs_act = comp_test_hs[['test_hs']].mean()
    hs_pred = comp_test_hs[['hs_per_month']].mean()

    percentage_error = (hs_pred - mse_for_hs) / hs_pred # error for the percentage error between the actual each month test hours and predict test hours for each month
    
    # error for the total test hours per project
    # ===================================================
    print (pre_only_project[['phase', 'total_hr_tradition', 'total_hr_pred', 'test_hs']].groupby('phase').mean())

    print (pre_only_project[['test_hs', 'total_hr_tradition', 'total_hr_pred']].mean())
    
    # ==================== plot ====================
    # ax.bar(pd.to_datetime(w_time['date'], format='%Y-%m-%d'), w_time['test_hs'])
    # ax.xaxis_date()

    w_time.set_index('date', inplace = True)
    
    w_time_new.set_index('new_date', inplace = True)

    # ax = w_time.plot(kind='bar', y = 'test_hs', figsize=(20, 9), color='#2ecc71', rot=0, alpha = .5)
    # ax = w_time_new.plot(kind='bar', y = 'hs_per_month', figsize=(20, 9), color='red', rot=0, ax = ax, alpha = .5)
    # ax.set_xticklabels(map(lambda x: line_format(x), w_time.index))

    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(90)
    #     tick.set_fontsize(6)
    # plt.savefig('work_load_svc.png')
    # # plt.show()

    
    