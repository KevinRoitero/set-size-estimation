#!/usr/bin/env python
# coding: utf-8


import numpy as np
import scipy as sp
from scipy import stats

def ANOVA2(alpha = 0.05, beta = 0.2, minD = 0.05, sigma2 = 0.2, m = 2, max_iter=1000):
    lambdaa = 4.86+3.584*np.sqrt(m-1)
    minDelta = (minD*minD)/(2*sigma2)
    n_approx = int(lambdaa / minDelta)

    large_enough = True
    i = 0
    n = n_approx+100
    while large_enough and i<max_iter:
        i+=1
        phiE = m*(n-1)
        phiA = m-1
        w = sp.stats.f.isf(alpha, phiA, phiE)
        cA = (phiA + 2 * n * minDelta)/(phiA + n * minDelta)
        phiA_star = ((phiA + n*minDelta)**2)/(phiA+2*n*minDelta)
        u_less_then = (np.sqrt(w/phiE)*np.sqrt(2*phiE-1)-np.sqrt(cA/phiA)*np.sqrt(2*phiA_star-1))/np.sqrt(cA/phiA - w/phiE)
        one_minus_beta_approx = 1-sp.stats.norm(loc=0, scale=1).cdf(u_less_then)

        large_enough = (1-beta) < one_minus_beta_approx
        n -= 1
    # add 
    n_recommended = n+2
    if i == max_iter:
        n_recommended = -1
    return (n_approx,n_recommended)
#ANOVA2(m=10)







