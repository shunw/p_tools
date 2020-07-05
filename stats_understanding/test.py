from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
import pylab
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sp.csv')

df['avg'] = (df['m1'] + df['m2'] + df['m3']) / 3

qqplot(df['avg'], line = 's')
plt.show()