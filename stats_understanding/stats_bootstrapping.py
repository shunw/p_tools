import numpy as np

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
