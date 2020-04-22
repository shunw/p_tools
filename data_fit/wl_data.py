import pandas as pd
import re
import joblib
import os
from functools import partial 
import calendar
from stats_anova import Anova_Bonferroni
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.dates as mdates
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor  
import warnings

warnings.filterwarnings("ignore")

'''
PURPOSE: prepare the data
purpose: try to predict the work load; 
assumption: 
0. before the work
    1. check if diff by project? <= with the page amount/ speed/ aio
    2. check if diff by phase? 
    3. check if diff by sf and aio? 
1. work load with time series
'''

# add mono/color, sf/aio info
project_ls = ['Arthur', 'Aurora', 'Bamboo', 'Birds', 'Canary', 'Dorado', 'Eyrie', 'Gawain', 'Hulk', 'KnightsMLK', 'Lark', 'Marlin', 'Moon', 'Nile', 'Pyramid', 'Seagull', 'Skyreach', 'Stars', 'Storm', 'Swan', 'Teton', 'Zenith', 'Annapuna', 'Asteroid', 'Buck', 'Clearwater', 'Dynasty', 'Gemini/Neptune', 'Homer', 'Mogami', 'Orion/Nebula', 'Sanya', 'Shimanto', 'Stella']

mono_sf_pro_ls = ['Arthur', 'Canary', 'Seagull', 'Dorado', 'Eyrie', 'KnightsMLK', 'Storm', 'Asteroid', 'Stars', 'Mogami', 'Sanya', 'Annapuna']
mono_aio_pro_ls = ['Bamboo', 'Gawain', 'Hulk', 'Lark', 'Marlin', 'Moon', 'Skyreach',  'Swan', 'Teton', 'Birds', 'Stella', 'Dynasty', 'Homer']

color_sf_pro_ls = ['Aurora', 'Nile', 'Shimanto', 'Buck']
color_aio_pro_ls = ['Pyramid', 'Zenith', 'Clearwater', 'Gemini/Neptune', 'Orion/Nebula']

detail_type_dict = {'c_sf': color_sf_pro_ls, 'c_aio': color_aio_pro_ls, 'm_sf': mono_sf_pro_ls, 'm_aio': mono_aio_pro_ls}

def stat_base_df(df, label, value): 
    '''
    input: df, label <- based on which to do the groupby cal; value <- on which col do the calculation
    '''
    # create the mean/ median/ var/ min/ max/ size df
    df = df[[label, value]].copy()
    # print (df.head())
    temp_df = df.groupby( [label], as_index = False).mean()
    temp_df.rename(columns={value: "mean"}, inplace = True)
    output_df = temp_df.copy()
    # print (output_df.head())
    temp_df = df.groupby( [label], as_index = False).var()
    output_df = output_df.merge(temp_df, left_on = label, right_on = label)
    output_df.rename(columns={value: "var"}, inplace = True)
    output_df['std'] =output_df['var'] ** .5
    
    temp_df = df.groupby( [label], as_index = False).size().reset_index()
    output_df = output_df.merge(temp_df, left_on = label, right_on = label)
    output_df.rename(columns={0: "n"}, inplace = True)
    return output_df


class Import_csv_2_pd(): 
    def __init__(self, fl_ls): 
        '''
        import: the filename list
        output: clean pd combined all the file data; saved as pkl file
        '''
        self.fl_ls = fl_ls # all the file name list
        # requirement: only remain the project/ months/ add item col

    def transfer_2_df(self): 
        n = 0

        for f in self.fl_ls: 
            temp_df = pd.read_csv(f)
            f_name = f.split('.')[0].split('/')[1]

            # get rid of all the empty col
            temp_df.dropna(how='all',axis=1, inplace = True)

            # only have the actual row
            temp_df = temp_df.loc[temp_df['item'] == 'Actual', ]

            # clear the col name
            col_name = list(temp_df.columns)
            col_dic = {c: c.split('.')[0] for c in col_name if '.1' in c}
            temp_df.rename(columns = col_dic, inplace = True)
            melt_col = col_dic.values()

            # melt col to row
            temp_df = temp_df.melt(id_vars = ['Projects', 'item'], value_vars = melt_col)

            # add the year col 
            temp_df['year'] = [f_name] * temp_df.shape[0]
            
            if n == 0: 
                self.df = temp_df.copy()
                n += 1
            else: 
                self.df = self.df.append(temp_df)
        return self.df

class Data_further_clear(): 
    '''
    further clear the data
    1. add project/ phase col
    2. define the project type (sf, aio)
    3. any chance to add the engine life, printer speed, adf speed? 
    '''
    def __init__(self, data): 
        '''
        input: df
        output: df
        '''
        self.df = data

    def _name_clean(self, row, except_ls): 
        # helper function to separate the name with project and phase
        prj = 'Projects'
        
        if row[prj] in except_ls[:-1]: 
            return row[prj], 'N/A'
        elif row[prj] == except_ls[-1]: 
            p_name, p_phase = row[prj].split(' ')
            return p_name.strip(), p_phase.strip()
        else: 
            splt_part = row[prj].split('_')
            part_qty = len(splt_part)
            if part_qty > 2: 
                return splt_part[0].strip(), splt_part[1].strip()
            try:     
                splt_part_2 = splt_part[0].split(' ')
                return splt_part_2[0].strip(), splt_part_2[1].strip()
            except: 
                return splt_part[0], 'N/A'
    def _add_type(self, row): 
        # helper function to add type
        pn = 'p_name'
        if row[pn] not in project_ls: 
            return 'N/A_N/A'
        else: 
            for k, v in detail_type_dict.items(): 
                if row[pn] in v: 
                    return k

    def add_proj_phase(self): 
        need_item = ['CPE Special Test', 'ISO Yield Tests', 'Cheetah', 'Tiger', 'Skyreach DE']
        # need_item = ['CPE Special Test', 'ISO Yield Tests', 'Cheetah', 'Tiger']
        drop_item = ['Scanner Team Special Test', 'Others', 'HW Team Special Test', 'On Leave', 'FW Test', 'Drop Test', 'Unallocated Time']
        
        # drop all some un-need items
        self.df = self.df[~self.df['Projects'].isin(drop_item)]

        
        # add project_name and phase col
        _name_clear_assit = partial(self._name_clean, except_ls = need_item)

        self.df['p_temp'] = self.df.apply(_name_clear_assit, axis = 1)
        self.df['p_name'] = self.df['p_temp'].apply(lambda x: x[0])
        self.df['phase'] = self.df['p_temp'].apply(lambda x: x[1])
        self.df.drop(['p_temp', 'item'], axis=1, inplace = True)

        # drop the N/A row
        self.df.dropna(subset = ['value'], inplace = True)

        # map the FY year to calendar year
        month_dict = dict((v, k) for k,v in enumerate(calendar.month_name))
        self.df['variable'] = self.df['variable'].apply(lambda x: month_dict[x])
        self.df['year'] = self.df[['year', 'variable']].apply(lambda x: str(int(x['year']) - 1) if int(x['variable']) >= 10 else x['year'], axis = 1)
        # print (self.df.head())

        # make time series
        
        self.df['date'] = self.df[['year','variable']].apply(lambda x: '{year}-{month}-{day}'.format(year = x['year'], month = x['variable'], day = 1), axis=1)
        self.df['date']=self.df['date'].apply(pd.to_datetime, format='%Y-%m-%d')
        # self.df.drop(['year', 'variable'], axis=1, inplace = True)
        self.df.rename(columns = {'value': 'test_hs', 'variable': 'month'}, inplace = True)
        # self.df['month'] = self.df['month'].apply(lambda x: month_dict[x])
        return self.df

    def add_type(self): 
        # add type like (mono, color, sf, aio)
        _add_type_assit = partial(self._add_type)
        self.df['t_type'] = self.df.apply(_add_type_assit, axis = 1)
        self.df['c_type'] = self.df['t_type'].apply(lambda x: x.split('_')[0])
        self.df['p_type'] = self.df['t_type'].apply(lambda x: x.split('_')[1])
        return self.df

class Work_load_Data():
    # this is to return the data
    def __init__(self): 
        df = joblib.load('df_data.pkl')
        clear = Data_further_clear(df)
        clear.add_proj_phase()
        self.df_prepared = clear.add_type()
        self.df_prepared = self.df_prepared[self.df_prepared['phase'].isin(['DE', 'MT', 'MP'])]

        # ============== remove unreasonable data ============== 
        self.df_prepared.drop(self.df_prepared[(self.df_prepared['phase'] == 'MP') & (self.df_prepared['p_name'] == 'Lark')].index, inplace = True)
        self.df_prepared.drop(self.df_prepared[(self.df_prepared['phase'] == 'MP') & (self.df_prepared['p_name'] == 'Stella')].index, inplace = True)
        
        
    def no_time_data(self): 
        '''
        only project information with the total test hours
        '''
        df_prepared_no_time = self.df_prepared.groupby(['phase', 'p_name', 't_type', 'c_type', 'p_type']).sum().reset_index()
        df_prepared_no_time.drop(['month'], axis = 1, inplace = True)
        
        df_month_qty = self.df_prepared.groupby(['p_name', 'phase']).size().reset_index()
        df_month_qty.rename(columns = {0:'month_qty'}, inplace = True)
        
        df_prepared_no_time = df_prepared_no_time.merge(df_month_qty, left_on =['p_name', 'phase'], right_on = ['p_name', 'phase'])

        df_prepared_no_time['tesths_per_month'] = df_prepared_no_time['test_hs']/ df_prepared_no_time['month_qty']
        
        return df_prepared_no_time

    

    def with_time_data(self): 
        '''
        only with time, no project information
        '''
        df_prepared_w_time = self.df_prepared.groupby(['year', 'month']).sum().reset_index()

        df_prepared_w_time['date'] = df_prepared_w_time[['year','month']].apply(lambda x: '{year}-{month}-{day}'.format(year = x['year'], month = x['month'], day = 1), axis=1)
        df_prepared_w_time['date'] = df_prepared_w_time['date'].apply(pd.to_datetime, format='%Y-%m-%d')

        return df_prepared_w_time
    
    def w_begin_time(self): 
        '''
        return the dataframe, with begin month start that project
        no each month detail for one project
        '''
        df_w_begin_time = self.df_prepared[['p_name', 'phase', 'date', 'c_type', 'p_type', 't_type']]
        remain_col = ['p_name', 'phase', 'c_type', 'p_type', 't_type']
        df_w_begin_time.sort_values(['p_name', 'date'], inplace = True)
        df_w_begin_time.drop_duplicates(remain_col, keep='first', inplace = True)
        
        return df_w_begin_time

    def w_begin_time_tradition(self): 
        '''
        return the project test hours with the traditional calculation/ estimation method
        '''
        df_basic = self.w_begin_time()
        df_basic.loc[df_basic['phase'] == 'MP', 'units'] = 9.0
        df_basic.loc[df_basic['phase'] == 'MT', 'units'] = 6.0
        df_basic.loc[df_basic['phase'] == 'DE', 'units'] = 3.0
        
        df_basic.loc[df_basic['p_type'] == 'sf', 'dur'] = 2.0 # 2 months
        df_basic.loc[df_basic['p_type'] == 'aio', 'dur'] = 3.0 # 2 months

        df_basic['total_hr_tradition'] = ((df_basic['units'] / 3 + .5) * 2 + .5 + .25 + .25) * 8 * 5 * 4 * df_basic['dur']
        return df_basic

    def proj_each_month(self): 
        '''
        return with each month detail information for one project
        the data purpose is to predict the each month detail for one project
        '''
        df_proj_each_month = self.df_prepared.sort_values(['p_name', 'phase', 'date']).copy()
        
        df_proj_each_month['month_incr']= df_proj_each_month.groupby(['p_name', 'phase']).cumcount() + 1
        need_col = ['p_name', 'phase', 'test_hs', 'month_incr', 'c_type', 'p_type', 't_type']
        return df_proj_each_month[need_col]


if __name__ == '__main__':
    pass
    # # =========== CSV data Import =============
    # name_ls = list(range(2013, 2021))
    # name_ls = ['.'.join([str(n), 'csv']) for n in name_ls]
    # dir_name = 'data'
    # name_ls = [os.path.join(dir_name, n) for n in name_ls]
    
    # deal_fl = Import_csv_2_pd(name_ls)
    # df = deal_fl.transfer_2_df()
    # joblib.dump(df, 'df_data.pkl')


    # # =========== DATA further Clear =============
    # # only check the three phase data
    # df = joblib.load('df_data.pkl')
    # clear = Data_further_clear(df)
    # clear.add_proj_phase()
    # df_prepared = clear.add_type()
    # df_prepared = df_prepared[df_prepared['phase'].isin(['DE', 'MT', 'MP'])]
    # # print (df_prepared.head())
    
    # df_prepared_no_time = df_prepared.groupby(['phase', 'p_name', 't_type', 'c_type', 'p_type']).sum().reset_index()
    # df_month_qty = df_prepared.groupby(['p_name', 'phase']).size().reset_index()
    # df_prepared_no_time.drop(['month'], axis = 1, inplace = True)
    
    # df_month_qty.rename(columns = {0:'month_qty'}, inplace = True)
    # df_prepared_no_time = df_prepared_no_time.merge(df_month_qty, left_on =['p_name', 'phase'], right_on = ['p_name', 'phase'])
    # df_prepared_no_time['tesths_per_month'] = df_prepared_no_time['test_hs']/ df_prepared_no_time['month_qty']
    # # df_prepared_no_time.to_csv('work_hours.csv',index = False)
    # # print (df_prepared_no_time.head())

    # df_prepared_w_time = df_prepared.groupby(['year', 'month']).sum().reset_index()
    # ['phase', 't_type', 'c_type', 'p_type', 'year', 'month']
    # print (df_prepared_w_time.head())
    # print (df_prepared_no_time.head())
    
    # # =========== Compare the MP/ MT/ DE phase Project Total test hours =============
    # '''
    # basic observation [no time]:  
    # to total test hours: [SIGNI]: type (sf, aio) diff; [NO SIGNI]: color diff; phase diff
    # to total test month: [SIGNI]: phase(MP-DE; MP-MT) diff; [NO SIGNI]: color diff; type(sf, aio);
    # to test hours per month: [SIGNI]: phase(DE-MP; DE-MT) diff; type (sf, aio) diff; [NO SIGNI]: color diff;

    # basic observation [w/ time]:  
    # to total test hours: [NO SIGNI]: month diff
    # '''
    # # check same phase different project
    # # phase = 'DE'
    # label = 'year'
    # data = 'test_hs'
    
    # # temp_data = df_prepared_no_time[[label, data]]
    # temp_data = df_prepared[[label, data]]

    # # # check different phase, only include the phase DE, MT, MP
    # # temp_data = df_prepared[df_prepared['phase'].isin(['DE', 'MT', 'MP'])]
    
    # anova_compare = Anova_Bonferroni(temp_data[data], temp_data[label])
    # anova_compare._anov_basic()
    # Fstat, Pr = anova_compare.anov_cal()
    # comp_result = anova_compare.pairwise_cmp(.05)

    # # print (Fstat, Pr)
    # # print (comp_result.loc[comp_result['j_significant'] == True, ])
    # print (comp_result)
    

    # mc = MultiComparison(temp_data[data], temp_data[label])
    # result = mc.tukeyhsd()

    # print (result)
    # print (mc.groupsunique)


    # # prepare the data for the f_oneway()
    # data_ls = []
    # label_ls = temp_data[label].unique()
    # for l in label_ls: 
    #     data_ls.append(temp_data.loc[temp_data[label] == l, data])

    # # print (stats.f_oneway(*data_ls))

    # # print (stats.kruskal(*data_ls))

    # # =========== Check Data =============

    # # use anova compare group
    # # =====================================
    # test_df = stat_base_df(df_prepared_w_time, 'year', 'test_hs')
    # print (test_df)
    # # test_df = stat_base_df(df_prepared_no_time, 't_type', 'test_hs')
    # # print (test_df)
    # # test_df = stat_base_df(df_prepared_no_time, 'p_type', 'test_hs')
    # # print (test_df)
    # # test_df = stat_base_df(df_prepared_no_time, 'phase', 'test_hs')
    # # print (test_df)

    # # view the test hours vs month and year
    # # =====================================
    # df_test = df_prepared_w_time.copy()
    # hr_year_month = df_test.pivot_table(index = ['month'], columns = 'year', aggfunc= lambda x: x)
    
    # hr_year_month.columns = hr_year_month.columns.droplevel().rename(None)
    # print (hr_year_month)

    # # view the test hours vs project
    # # =====================================
    # print (df_prepared_no_time.head())
    # temp_df_4_pivt = df_prepared_no_time.loc[df_prepared_no_time['c_type'] == 'm', ['p_name', 'phase', 'test_hs']] # 'month_qty', 
    # pro_phase = temp_df_4_pivt.pivot_table(index = ['p_name'], columns = 'phase', aggfunc = lambda x: x)
    # print (pro_phase)
    
    # # =========== Data Box Plot=============
    # boxplot = df_prepared_w_time.boxplot(column = ['test_hs'], by = 'year')
    # plt.show()

    # # =========== Scatter/ Line Plot=============
    # # need to re-structure dataframe due to the time series 
    # df_time_plot = df_prepared_w_time.copy()

    # df_time_plot['date'] = df_time_plot[['year','month']].apply(lambda x: '{year}-{month}-{day}'.format(year = x['year'], month = x['month'], day = 1), axis=1)
    # df_time_plot['date'] = df_time_plot['date'].apply(pd.to_datetime, format='%Y-%m-%d')
    

    # # plot the line, different year, compare month
    # # ===============================================
    # df_time_plot['month_1'] = df_time_plot['date'].dt.strftime('%b')
    
    # level_d = df_time_plot['year'].unique()
    # colormap = cm.viridis
    # colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, .9, len(level_d))]

    # for ind, y in enumerate(level_d): 
    #     y1 = df_time_plot.loc[df_time_plot['year'] == y, ['month_1', 'test_hs']]
        
    #     if ind == 0:
    #         ax = y1.plot(x = 'month_1', y = 'test_hs', c = colorlist[ind], label = y)
    #     else:
    #         ax = y1.plot(x='month_1', y='test_hs', ax=ax, c = colorlist[ind], label = y)
    # ax.legend(bbox_to_anchor = (1, 1))
    # plt.show()

    # # # plot the line, time series
    # # # ===============================================
    # # df_time_plot.plot(x = 'date', y = 'test_hs')
    # x = pd.to_datetime(df_time_plot['date'], format='%Y-%m-%d')
    # plt.scatter(x, df_time_plot['test_hs'])
    # plt.show()

    # # =========== HIST Plot=============
    # # print (df_prepared_no_time.head())
    # df_prepared_no_time.loc[df_prepared_no_time['p_type'] == 'sf', ].hist('test_hs', bins = 50)
    # plt.show()

    # # ============LINEAR REG==================
    # # print (df_prepared_no_time.head())
    
    # lr_fit = LinearRegression()
    # y = df_prepared_no_time['test_hs']
    # X = pd.get_dummies(df_prepared_no_time[['phase', 'c_type', 'p_type']])
    # lr_fit.fit(X, y)
    # # print (X.head())
    # X_test = np.array([[1, 0, 0, 0, 1, 0, 1]])
    # # phase_DE  phase_MP  phase_MT  c_type_c  c_type_m  p_type_aio  p_type_sf
    # print (lr_fit.predict(X_test))
    # print (lr_fit.predict(np.array([[0, 1, 0, 0, 1, 0, 1]])))
    # print (lr_fit.predict(np.array([[0, 0, 1, 0, 1, 0, 1]])))
    # tree = DecisionTreeRegressor()
    # tree.fit(X, y)
    # print (tree.score(X, y))



    