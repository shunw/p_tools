import warnings
import numpy as np
import scipy.stats as st
import statistics as stat
import matplotlib.pyplot as plt
import pandas as pd

class data_fit_plot(object):
    def __init__(self, data, fit_type = 'best'):
        self.data = data

        self.mu = stat.mean(self.data)
        self.std = stat.stdev(self.data)

        self.fit_type = fit_type
        self.fit_type_dict = {'normal': st.norm, 'chi': st.chi}

        bins = 50
        self.y, x = np.histogram(self.data, bins=bins, density=True)
        self.x = (x + np.roll(x, -1))[:-1] / 2.0

        self.to_check_distribution = [        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]
    
    def find_best_distribution(self):
        self.best_distribution = st.norm
        self.best_params = (0.0, 1.0)
        self.best_sse = np.inf

        for d in self.to_check_distribution:
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                
                    params = d.fit(self.data)
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    pdf = d.pdf(self.x, loc = loc, scale = scale, *arg)
                    sse = np.sum(np.power(self.y - pdf, 2.0))

                    if self.best_sse > sse > 0:
                        self.best_distribution = d
                        self.best_params = params
                        self.best_sse = sse
                        
            
            except Exception:
                pass
        
        return self.best_sse, self.best_distribution.name, self.best_params



    def data_fit(self):
        # Fit a normal distribution to the data:
        self.mu, self.std = self.fit_type.fit(data)

    def make_pdf(self, dist, params, size = 10000):
        """Generate distributions's Probability Distribution Function """

        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Get sane start and end points of distribution
        start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
        end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

        # Build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = dist.pdf(x, loc=loc, scale=scale, *arg)
        pdf = pd.Series(y, x)

        
        return pdf


    def data_plot(self):
        # Plot the histogram.
        plt.hist(self.data, bins=25, normed = True, alpha=0.6, color='g')

        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        
        if self.fit_type == 'best': 
            p = self.make_pdf(self.best_distribution, self.best_params)
            p.plot(lw = 2, label = 'PDF')
            d_name = self.best_distribution.name
        else: 
            p = self.fit_type.pdf(x, self.mu, self.std)
            plt.plot(x, p, color = 'k', linewidth=2)
            d_name = self.fit_type.name
        title = "Fit {name} results: mu = {mu:.2f},  std = {std:.2f}".format(name = d_name, mu = self.mu, std = self.std)
        plt.title(title)
        plt.show()

    def final_run(self):
        if self.fit_type in self.fit_type_dict.keys():
            # self.fit_type_dict = {'normal': st.norm}
            self.fit_type = self.fit_type_dict[self.fit_type]
            self.data_fit()
            
        
        else: 
            self.find_best_distribution()
        
        self.data_plot()

    

if __name__ == '__main__': 
    data = st.norm.rvs(10.0, 2.5, size=500)
    data_fit = data_fit_plot(data, 'chi')
    data_fit.final_run()

    '''
    current study the chi distribution
    '''