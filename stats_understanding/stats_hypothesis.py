import numpy as np
import scipy.stats as st
from scipy.stats import ttest_ind, ttest_ind_from_stats

class CompareTwoSamples_Mean(): 
    def __init__(self, sample_1, sample_2, var_1 = None, var_2 = None): 
        '''
        sample_1 and sample_2 are np.array
        var_1 and var_2 is float value
        output includes: p_value, p, df, loc, dividend / scale
        '''
        self.sample_1, self.sample_2 = sample_1, sample_2
        
        self.mean_1, self.mean_2 = np.mean(self.sample_1), np.mean(self.sample_2)
        
        self.n_1, self.n_2 = len(self.sample_1), len(self.sample_2)

        if (var_1 is not None) and (var_2 is not None): 
            self.var_1, self.var_2 = var_1, var_2
            self.var_known = True

        elif self.n_1 >= 30 and self.n_2 >= 30: 
            self.var_1, self.var_2 = np.var(self.sample_1), np.var(self.sample_2)
            self.var_known = True
            
        else: 
            self.var_1, self.var_2 = np.var(self.sample_1), np.var(self.sample_2)
            self.var_known = False
        
        
    
    def get_p(self): 
        
        self.df = 0

        if (self.n_1 >= 30 and self.n_2 >= 30) or self.var_known: 
            # z test due to sample size bigger than 30
            # or var known though size is not bigger than 30
            self.scale = (self.var_1/ self.n_1 + self.var_2/ self.n_2) ** .5
            self.loc = (self.mean_1 - self.mean_2)
            self.p_value = self.loc/ self.scale
            self.p = st.norm.pdf(self.p_value)
            return self.p_value, self.p, self.df, self.loc, self.scale
        
        else:
            if self.var_1 == self.var_2: 
                # var1 == var2 unknown
                Sp = ((self.var_1 * (self.n_1 - 1) + self.var_2 * (self.n_2 - 1)) / (self.n_1 + self.n_2 - 2)) ** .5
                self.scale = Sp * ((1.0/ self.n_1 + 1.0/ self.n_2) ** .5)
                self.df = self.n_1 + self.n_2 - 2

            elif self.var_1 != self.var_2 and self.n_1 == self.n_2:
                # var1 != var2 and both unknown and n1 == n2
                self.scale = (self.var_1/ self.n_1 + self.var_2 / self.n_2) ** .5
                self.df = self.n_1 + self.n_2 - 2

            elif self.var_1 != self.var_2 and self.n_1 != self.n_2:
                # var1 != var2 and both unknown and n1 != n2
                self.scale = (self.var_1/ self.n_1 + self.var_2 / self.n_2) ** .5
                part_1 = self.var_1 / self.n_1
                part_2 = self.var_2 / self.n_2
                self.df = (part_1 + part_2) ** 2/ (part_1 ** 2 / (self.n_1 - 1) + part_2 ** 2 / (self.n_2 - 1))
            else: 
                print ('the situation is not listed.')
            self.loc = (self.mean_1 - self.mean_2)
            self.p_value = self.loc / self.scale
            self.p = st.t.pdf(self.p_value, self.df)

            return self.p_value, self.p, self.df, self.loc, self.scale

    def get_interval(self, alpha): 
        if self.var_known: 
            return st.norm.interval(alpha, self.loc, self.scale)
        else: 
            return st.t.interval(alpha,self.df, self.loc, self.scale)