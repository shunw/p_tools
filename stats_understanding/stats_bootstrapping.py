import numpy as np
import pandas as pd

class Bootstrapping(): 
    def __init__(self, sample, b = 10000): 
        self.b = b
        self.sample = sample

    def simulate(self): 
        self.b_resample = None

        for _ in range(self.b): 
            temp = np.random.choice(self.sample, len(self.sample), replace = True) 
            
            if self.b_resample is None: 
                self.b_resample = temp
            else: 
                self.b_resample = np.vstack((self.b_resample, temp))
            
        return self.b_resample
    
    def b_means(self): 
        return np.mean(self.b_resample, axis = 1)

    def b_median(self): 
        return np.median(self.b_resample, axis = 1)

class Bootstrapping_hypothesis():
    def __init__(self, sample_data, sample_label, b = 10000): 
        '''
        input: sample_data/ sample could be the pandas series
        data creation: sample_n * b
        '''
        self.b = b
        self.sample_data = sample_data
        self.sample_label = sample_label

    def bs_simulate(self): 
        self.b_resample = None
        self.base_df = pd.DataFrame({'group': self.sample_label, 're_sample': self.sample_data})
        a, b = self.base_df.groupby(['group']).sum()['re_sample']
        base = a - b

        b_sample_data = []

        for _ in range(self.b): 
            temp = np.random.choice(self.sample_data, len(self.sample_data), replace = True) 
            
            temp_df = pd.DataFrame({'group': self.sample_label, 're_sample': temp})
            a, b = temp_df.groupby(['group']).sum()['re_sample']
            b_sample_data.append(abs(a - b))
        # print (b_sample_data)
        
        obs_qty = (np.array(b_sample_data) >= base).sum()
        return base, obs_qty/self.b
        
        

if __name__ == '__main__': 

    df = pd.read_excel('CFT_data_2.xlsx', header = 0)
    # x_col = 'Media Loading' 
    # x_col = 'Media Type Setting' 
    # x_col = 'Tray Side Guide' 
    # x_col = 'Tray Back Guide' 
    x_col = 'Media Fanning'
    occ_col = 'Jam (Occurances)'
    img = 'sum(# of Page)'
    df_short = df[[x_col, img, occ_col]]
    
    bs = Bootstrapping_hypothesis(df_short[occ_col], df_short[x_col])
    print (bs.bs_simulate())
    

