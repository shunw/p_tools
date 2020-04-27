from wl_data import Work_load_Data, Training_Testing_data_get

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
from sklearn.model_selection import train_test_split

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
def prepare_df_totaly_hours(df_projects_w_months): 
    '''
    input: dataframe (with no date information, but list all the projects hours for each month)
    return: dataframe with X, y_month_qty, y_test_hs_total
        X has the following dealing. 

        1. add month_qty col to calculate how many month execute the test
        2. remove the month_incr col
        3. groupby and sum each projects total testing hours
    '''
    X_need_col = ['p_name', 'phase', 'c_type', 'p_type', 't_type']
    df = df_projects_w_months.copy()
    df.drop(columns = ['month_incr'], inplace = True)
    
    df_month_qty = df.groupby(['p_name', 'phase']).size().reset_index()
    df_month_qty.rename(columns = {0:'month_qty'}, inplace = True)
    
    df = df.groupby(X_need_col).sum().reset_index()

    df = df.merge(df_month_qty, left_on =['p_name', 'phase'], right_on = ['p_name', 'phase'])
    X_need_col.remove('p_name')
    X_need_col.remove('t_type')
    return df[X_need_col], df[['month_qty']], df[['test_hs']]

def prepare_df_each_month_hours(df_projects_w_months): 
    '''
    input: dataframe (with no date information, but list all the projects hours for each month)
    return: dataframe with X, y_test_hs_each_month
        X has the following dealing. 
        just choose suitable col
    '''
    df = df_projects_w_months.copy()
    X_need_col = ['phase', 'month_incr', 'c_type', 'p_type']

    return df[X_need_col], df[['test_hs']]
    
if __name__ == '__main__': 
    data_m = Training_Testing_data_get().all_data()
    no_time = data_m.no_time_data() # no time, actual test hours
    w_time = data_m.with_time_data()
    w_pro_time = data_m.w_begin_time()
    pro_w_each_m, pro_w_each_m_addT = data_m.proj_each_month()

    no_time_trd = data_m.w_begin_time_tradition() # traditional calculation
    

    data_new = Training_Testing_data_get()
    train_no_time, train_w_time, test_no_time, test_w_time = data_new.training_test_split()
    # ================ prepare the training data ================ 

    # only with project, no each month
    # =============================================================
    # y_col = 'month_qty' # ['test_hs', 'month_qty', 'tesths_per_month']
    # col_no_y = list(no_time.columns)
    # col_no_y.remove(y_col)
    # X = no_time[col_no_y]
    # data_pre_transfer = CombineAttributes(est_target = y_col)
    # data_pre_transfer.fit_transform(X)


    # # print (X.head())
    
    X_4_train_hs_total, y_train_month_qty, y_train_hs_total = prepare_df_totaly_hours(train_no_time)
    X_4_train_hs_total_dummy = pd.get_dummies(X_4_train_hs_total)
    
    scaler_hs_total = StandardScaler()
    X_4_train_hs_total_dummy_scaled = scaler_hs_total.fit_transform(X_4_train_hs_total_dummy)
    
    # # with project, with each month
    # # =============================================================
    # y_col_m = 'test_hs'

    # col_no_y_m = list(pro_w_each_m.columns)
    # col_no_y_m.remove(y_col_m)
    
    # Xm = pro_w_each_m[col_no_y_m]
    
    # # print (pro_w_each_m.shape)
    # data_pre_transfer_m = CombineAttributes(est_target = 'test_hs', w_month = True)
    # data_pre_transfer_m.fit_transform(Xm)
    # # print (Xm.head())
    # # print (pro_w_each_m.shape)
    # # print (pro_w_each_m.head())
    # print (Xm.head())
    # Xm = pd.get_dummies(Xm)

    # scaler_m = StandardScaler()
    # Xm_scale = scaler_m.fit_transform(Xm)
    
    # ym = pro_w_each_m[[y_col_m]]

    X_4_train_hs_each_month, y_train_hs_each_month = prepare_df_each_month_hours(train_no_time)
    X_4_train_hs_each_month_dummy = pd.get_dummies(X_4_train_hs_each_month)
    
    scaler_hs_each_month = StandardScaler()
    X_4_train_hs_each_month_dummy_scaled = scaler_hs_each_month.fit_transform(X_4_train_hs_each_month_dummy)


    # ==================== get two kinds of dummy data ====================
    test_hs_total, test_hs_total_dum, test_hs_each_month, test_hs_each_month_dum = get_dummy_data()

    # ==================== module and prediction ====================
    linear = LinearRegression()
    
    tree = DecisionTreeRegressor()

    rf = RandomForestRegressor(n_estimators = 100)
    ef = ExtraTreesRegressor(n_estimators = 100)
    grbt = GradientBoostingRegressor(n_estimators= 100)

    svr = SVR(gamma= 'scale')

    test_alg = clone(grbt)
    test_alg_2 = clone(grbt)
    test_alg_m = clone(grbt)
    
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
    
    # # with project, with each month
    # # =============================================================
    # test_df_dum_m_scale = scaler_m.transform(test_df_dum_m)

    # test_alg_m.fit(Xm_scale, ym)
    # y_pred_m = test_alg_m.predict(test_df_dum_m_scale)
    # test_df_m['test_hs_m_pred'] = y_pred_m
    
    # # # ==================== check learning curve ====================
    # # plot_learning_curves(test_alg, X, y)
    # # plt.show()

    # # plot_learning_curves(test_alg_m, Xm_scale, ym)
    # # plt.show()
    
    # # # ================== HIST PLOT ==================
    # # plt.hist(y_pred, bins = 50, alpha = .5, label = 'pred')
    # # plt.hist(y.to_numpy(), bins = 50, alpha = .5, label = 'act')
    # # plt.legend(loc = 'upper right')
    # # plt.show()
   
    # # ================== BAR PLOT TIME SERIES==================

    # # ==================== combine actual pro w/ predition ====================

    # pre_w_project = w_pro_time.merge(test_df, left_on = ['phase', 'c_type', 'p_type'], right_on = ['phase', 'c_type', 'p_type'], )
    
    # pre_only_project = pre_w_project.copy() # only project and its total test hours

    # # with time, add the test hours / month (avg)
    # # ===================================================
    # pre_w_project = pre_w_project.loc[pre_w_project.index.repeat(pre_w_project['month_qty'])]
    # # pre_w_project['month_incr'] = [1] * pre_w_project.shape[0]
    # pre_w_project['for_month_incr'] = pre_w_project['p_name'] + pre_w_project['phase']
    # pre_w_project['month_incr']= pre_w_project.groupby('for_month_incr').cumcount()
    
    # pre_w_project['new_date'] = pre_w_project.apply(_add_month, axis = 1)
    
    # need_col = ['new_date', 'hs_per_month']
    # w_time_pred = pre_w_project[need_col]
    
    # w_time_new = w_time_pred.groupby('new_date').sum().reset_index()

    # # with no time, only project
    # # ===================================================
    # pre_only_project['total_hr_pred'] = pre_only_project['month_qty'] * pre_only_project['hs_per_month']
    
    # p_need_col = ['p_name', 'phase', 'total_hr_tradition', 'total_hr_pred', 'test_hs']

    # pre_only_project = pre_only_project.merge(no_time_trd, left_on = ['p_name', 'phase'], right_on = ['p_name', 'phase'])
    # pre_only_project = pre_only_project.merge(no_time, left_on = ['p_name', 'phase'], right_on = ['p_name', 'phase'])[p_need_col]
    
    # # with project, with each month
    # # =============================================================
    # pre_w_project_m = pro_w_each_m.merge(test_df_m, left_on = ['phase', 'month_incr', 'c_type', 'p_type'], right_on = ['phase', 'month_incr', 'c_type', 'p_type'])
    
    # # print (pre_w_project_m.head())
    # # print (pro_w_each_m_addT.head())
    # ready_for_bar_compare = pre_w_project_m.merge(pro_w_each_m_addT, left_on = ['p_name', 'phase', 'test_hs', 'month_incr'], right_on = ['p_name', 'phase', 'test_hs', 'month_incr'])
    # actual_m = ready_for_bar_compare[['test_hs', 'date']]
    # actual_m = actual_m.groupby(['date']).sum().reset_index()

    # predict_m = ready_for_bar_compare[['test_hs_m_pred', 'date']]
    # predict_m = predict_m.groupby(['date']).sum().reset_index()
    # predict_m.rename(columns = {'test_hs_m_pred': 'hs_per_month', 'date': 'new_date'}, inplace = True)
    # # print (actual_m.head())
    # # print (predict_m.head())

    # # ==================== cal_errors ====================

    # # error for the test hours per month (avg)
    # # ===================================================
    # comp_test_hs = w_time.merge(w_time_new, left_on = 'date', right_on = 'new_date')
    # mse_for_hs = mean_squared_error(comp_test_hs[['test_hs']], comp_test_hs[['hs_per_month']]) ** .5 # mean squared error
    # # print (mse_for_hs)
    
    # hs_act = comp_test_hs[['test_hs']].mean()
    # hs_pred = comp_test_hs[['hs_per_month']].mean()
    # # print (comp_test_hs[['test_hs', 'hs_per_month']].mean())

    # # percentage_error = ? # error for the percentage error between the actual each month test hours and predict test hours for each month
    
    # temp_m_for_compare = ready_for_bar_compare.groupby(['date']).sum().reset_index()
    # # print (temp_m_for_compare[['test_hs', 'test_hs_m_pred']].mean())
    # # print (mean_squared_error(actual_m[['test_hs']], predict_m[['hs_per_month']]) ** .5)

    # # # error for the total test hours per project
    # # ===================================================
    # # print (pre_only_project[['phase', 'total_hr_tradition', 'total_hr_pred', 'test_hs']].groupby('phase').mean())

    # # print (pre_only_project[['test_hs', 'total_hr_tradition', 'total_hr_pred']].mean())

    # a = pre_w_project_m.groupby(['p_name', 'phase']).sum().reset_index()
    # # print (a.groupby(['phase']).mean())
    # # print (a[['test_hs', 'test_hs_m_pred']].mean())
    
    # # # ==================== plot ====================
    # # print (w_time.head())
    # # print (w_time_new.head())

    # # plot_month_compare_bar(w_time, w_time_new) # predict avg month
    # # plot_month_compare_bar(actual_m, predict_m, save_fig= False) # predict each month
    
    
    