from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
import pylab
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math

df = pd.read_csv('sp.csv')

total_data = df[['m1', 'm2', 'm3']].stack()

mu = total_data.mean()
var = total_data.var()
std = math.sqrt(var)
x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
plt.plot(x, norm.pdf(x, mu, std), 'rx')
plt.plot(x, norm.cdf(x, mu, std), 'r')
plt.plot(x, norm.pdf(x, mu, std/(math.sqrt(3))), 'bx')
plt.plot(x, norm.cdf(x, mu, std/(math.sqrt(3))), 'b')
plt.plot(x, 1 - (1 - norm.cdf(x, mu, std/(math.sqrt(3)))) * (1 - norm.cdf(x, mu, std)), 'k')
plt.show()

