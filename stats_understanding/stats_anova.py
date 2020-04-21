import pandas as pd
from scipy import stats
import numpy as np
from itertools import combinations

def combine(arr, s): 
    return list(combinations(arr, s))

class Anova_Bonferroni(): 
    def __init__(self, all_data, all_label): 
        '''
        deal as the turkey in stats model, one is the data_col, the other is the label_col
        dataset defined as the dataframe for not
        dataset should be cleaned, no nan or others
        '''
        self.all_data = all_data
        self.all_label = all_label
        self.df = pd.DataFrame({'data': self.all_data, 'label': self.all_label})
        self.total_mean = self.df['data'].mean()
        
    def _anov_basic(self): 
        self.anov_df = self.df.groupby( [ "label"], as_index = False).mean()
        self.anov_df.rename(columns={"data": "mean"}, inplace = True)
        
        temp = self.df.groupby( [ "label"], as_index = False).var()
        self.anov_df = self.anov_df.merge(temp, left_on = 'label', right_on = 'label')
        self.anov_df.rename(columns={"data": "var"}, inplace = True)
        self.anov_df['var2'] = self.anov_df['var'] ** 2
        
        temp = self.df.groupby( [ "label"], as_index = False).size().reset_index()
        self.anov_df = self.anov_df.merge(temp, left_on = 'label', right_on = 'label')
        self.anov_df.rename(columns={0: "n"}, inplace = True)

        
    def anov_cal(self): 
        self._anov_basic()
        self.grp_num = self.anov_df.shape[0]

        self.SSb = ((self.anov_df['mean'] - self.total_mean) ** 2 * self.anov_df['n']).sum()
        self.dfb = self.grp_num-1 # == dfn
        self.MSb = self.SSb / self.dfb # variation between group/ explainable/ Mean Square Between
        
        self.SSw = (self.anov_df['var'] * (self.anov_df['n'] - 1)).sum()
        self.dfw = self.df.shape[0] - self.anov_df.shape[0] # == dfd
        self.MSw = self.SSw / self.dfw # variation within group/ unexplainable/ Mean Square Within
        
        self.Fstat = self.MSb/ self.MSw
        
        self.Pr = 1 - stats.f.cdf(self.Fstat, self.dfb, self.dfw)
        return self.Fstat, self.Pr

    def pairwise_cmp(self, alpha): 
        # use bonferroni correction method
        # suppose for the two side 
        self.anov_cal()
        
        pairwise_lower = []
        pairwise_upper = []
        comp_1 = []
        comp_2 = []
        judge = []

        set_num = 2
        ind_array = list(range(self.anov_df.shape[0]))
        all_combin_ls = combine(ind_array, set_num)

        pairwise_n = len(all_combin_ls)
        t_val = stats.t.ppf(1 - alpha/ pairwise_n/ 2, self.dfw) 
        
        for i, j in all_combin_ls: 
            se = (self.MSw / self.anov_df['n'][i] + self.MSw/ self.anov_df['n'][j]) ** .5
            mean_diff = self.anov_df['mean'][i] - self.anov_df['mean'][j]
            lower = mean_diff - t_val * se
            upper = mean_diff + t_val * se
            pairwise_lower.append(lower)
            pairwise_upper.append(upper)
            comp_1.append(self.anov_df['label'][i])
            comp_2.append(self.anov_df['label'][j])
            if lower * upper > 0: judge.append(True)
            else: judge.append(False)
        pairwise_df = pd.DataFrame({'comp_1': comp_1, 'comp_2': comp_2, 'lower': pairwise_lower, 'upper': pairwise_upper, 'j_significant': judge})
        return pairwise_df
        

if __name__ == '__main__': 

    df = pd.read_csv('DietWeigthLoss.csv')
    df['WeightLoss'] = df['WeightLoss\tDiet'].apply(lambda x: float(x.split()[0]))
    df['Diet']= df['WeightLoss\tDiet'].apply(lambda x: x.split()[1])
    df=df[['WeightLoss', 'Diet']]

    wtl = 'WeightLoss'
    dt = 'Diet'

    anb = Anova_Bonferroni(df[wtl], df[dt])
    anb._anov_basic()
    anb.anov_cal()
    print (anb.pairwise_cmp(.05))

    # Notice, for some complex one, the Fstat and Pr value may different from the f one-way in the Scipy. 
    # the F stat is similar 
    # the Pr is quite different, not sure if this is cause by the df, but both are close to 0




