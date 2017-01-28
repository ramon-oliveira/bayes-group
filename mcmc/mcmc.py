# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns

from scipy.stats import norm


def sampler(data, samples=4, mu_init=.5, proposal_width=.5, mu_prior_mu=1.7, mu_prior_sd=3.0, burnin=100, thin=10):
    mu_current = mu_init
    global acceptRatio
    acceptRatio = 0
    posterior = [mu_current]
    for i in range(samples):
        for j in range(thin):
            # suggest new position
            mu_proposal = norm(mu_current, proposal_width).rvs()
    
            # Compute likelihood by multiplying probabilities of each data point
            loglikelihood_current = np.log(norm(mu_current, 1).pdf(data)).sum()
            loglikelihood_proposal = np.log(norm(mu_proposal, 1).pdf(data)).sum()
            
            # Compute prior probability of current and proposed mu        
            prior_current = np.log(norm(mu_prior_mu, mu_prior_sd).pdf(mu_current))
            prior_proposal = np.log(norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal))
            
            p_current = loglikelihood_current + prior_current
            p_proposal = loglikelihood_proposal + prior_proposal
            
            # Accept proposal?
            p_accept = np.exp(p_proposal - p_current)
            
            # Usually would include prior probability, which we neglect here for simplicity
            accept = np.random.rand() < p_accept
            
            if accept:
                # Update position
                mu_current = mu_proposal
                acceptRatio += 1;
            
        posterior.append(mu_current)
        
        
    acceptRatio = acceptRatio/(thin*samples)    
    
    return posterior[burnin:]

def calc_posterior_analytical(data, x, mu_0, sigma_0):
    sigma = 1.
    n = len(data)
    mu_post = (mu_0 / sigma_0**2 + data.sum() / sigma**2) / (1. / sigma_0**2 + n / sigma**2)
    sigma_post = (1. / sigma_0**2 + n / sigma**2)**-1
    return norm(mu_post, np.sqrt(sigma_post)).pdf(x)
    
np.random.seed(123)

data = np.array([1.7 , 1.85, 1.82, 1.75])

posterior = sampler(data,1000,1.7,1)
media = np.mean(posterior)
plt.figure(1)
b = plt.plot(posterior)

plt.figure(2)

sns.distplot(posterior, label='estimated posterior')
x = np.linspace(0, 3, 500)
post = calc_posterior_analytical(data, x, 1.7, 3)
plt.subplot().plot(x, post, 'g', label='analytic posterior')
_ = plt.subplot().set(xlabel='mu', ylabel='belief');
plt.subplot().legend();

