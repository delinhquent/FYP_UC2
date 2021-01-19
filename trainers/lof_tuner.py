import numpy as np

from collections import defaultdict
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import nct

import tqdm

"""
Credits to : https://github.com/vsatyakumar/automatic-local-outlier-factor-tuning/blob/master/lof_tuner.py
The code can be found as mentioned in the link above. This script has been modified to include codes which are relevant for this project.
"""

"""
Algorithm : Tuning algorithm for LOF
1: training data X ∈ R
n×p
2: a grid of feasible values gridc
for contamination c
3: a grid of feasible values gridk
for neighborhood size k
4: for each c ∈ gridc do
5: for each k ∈ gridk do
6: set Mc,k,out to be mean log LOF for the bcnc outliers
7: set Mc,k,in to be mean log LOF for the bcnc inliers
8: set Vc,k,out to be variance of log LOF for the bcnc outliers
9: set Vc,k,in to be variance of log LOF for the bcnc inliers
10: set Tc,k = √
Mc,k,out−Mc,k,in
1
bcnc (Vc,k,out+Vc,k,in)
11: end for
12: set Mc,out to be mean Mc,k,out over k ∈ gridk
13: set Mc,in to be mean Mc,k,in over k ∈ gridk
14: set Vc,out to be mean Vc,k,out over k ∈ gridk
15: set Vc,in to be mean Vc,k,in over k ∈ gridk
16: set ncpc = √
Mc,out−Mc,in
1
bcnc (Vc,out+Vc,in)
17: set dfc = 2bcnc − 2
18: set kc,opt = arg maxk Tc,k
19: end for
20: set copt = arg maxc P(Z < Tc,kc,opt ; d fc
, ncpc), where the random variable Z follows a noncentral
t distribution with dfc degrees of freedom and ncpc noncentrality parameter
"""


class LOFAutoTuner(object):
    def __init__(self, n_samples = 500, c_max = 50, k_max = 1, data = None):
    
        if data is None:
            self.n_samples = n_samples
            print("Input 'data', array-like, shape : (n_samples, n_features).")
        else:
            self.data = data
            self.n_samples = self.data.shape[0]
        
        self.eps = 1e-8
        self.c_max = c_max
        self.k_max = k_max
        self.c_steps = 100
        self.k_grid = np.arange(1,self.k_max + 1) #neighbors
        self.c_grid = np.linspace(0.005, self.c_max, self.c_steps) #contamination
    
    def run(self):
        self.collector = []
        #main op
        for contamination in tqdm.tqdm(self.c_grid):
            samps = int(contamination * self.n_samples)

            #init running metrics
            running_metrics = defaultdict(list)
            for k in self.k_grid:
                clf = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
                clf.fit_predict(self.data)
                X_scores = np.log(- clf.negative_outlier_factor_)
                t0 = X_scores.argsort()

                x_out = X_scores[t0[-samps:]]
                x_in = X_scores[t0[:samps]]

                mc_out = np.mean(x_out)
                mc_in = np.mean(x_in)
                vc_out = np.var(x_out)
                vc_in = np.var(x_in)
                Tck = self.similar_formula(mc_out=mc_out, mc_in=mc_in,samps=samps,vc_out=vc_out,vc_in = vc_in)

                running_metrics['tck'].append(Tck)
                running_metrics['mck_out'].append(mc_out)
                running_metrics['mck_in'].append(mc_in)
                running_metrics['vck_in'].append(vc_in)
                running_metrics['vck_out'].append(vc_out)

            largest_idx = np.array(running_metrics['tck']).argsort()[-1]
            mean_mc_out = np.mean(running_metrics['mck_out'])
            mean_mc_in = np.mean(running_metrics['mck_in'])
            mean_vc_out = np.mean(running_metrics['vck_out'])
            mean_vc_in = np.mean(running_metrics['vck_in'])

            #ncpc - non-centrality parameter
            ncpc = self.similar_formula(mc_out=mean_mc_out, mc_in=mean_mc_in,samps=samps,vc_out=mean_vc_out,vc_in = mean_vc_in)

            #dfc - degrees of freedom
            dfc = (2*samps) - 2

            Z = nct(dfc, ncpc) #non-central t-distribution
            Kopt = self.k_grid[largest_idx]
            Topt = running_metrics['tck'][largest_idx]
            Z = Z.cdf(Topt)
            self.collector.append([Kopt, Topt, Z, contamination])      

        
        self.tuned_params = self.find_best_param()
        
        return self.tuned_params
    
    def find_best_param(self):
        max_cdf = 0.
        self.tuned_params = {}
        for v in self.collector:
            Kopt, Topt, Z, contamination = v
            if Z > max_cdf:
                max_cdf = Z

            if max_cdf == Z:
                self.tuned_params['k'] = Kopt
                self.tuned_params['c'] = contamination
        print("\nTuned LOF Parameters : {}".format(self.tuned_params))

        return self.tuned_params
    
    def similar_formula(self,**kwargs):
        return (kwargs['mc_out'] - kwargs['mc_in'])/np.sqrt((self.eps + ((1/kwargs['samps'])*(kwargs['vc_out'] + kwargs['vc_in']))))