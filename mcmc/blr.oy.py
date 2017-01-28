# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:59:18 2016

@author: amenegola
"""

from pymc import *
import pylab as plt
import numpy as np
import seaborn


A_true = 10.0
B_true = 3.0
X = np.linspace(-1, 1, 1000) #np.random.uniform(low=-1, high=1, size=[10])
Y_true = X*A_true + B_true + np.random.normal(size=[1000])

A = Normal('A', mu=0, tau=0.1)
B = Normal('B', mu=0, tau=0.1)
tau = Uniform('tau', 0, 1000)
Y = Normal('Y', mu=A*X+B, tau=tau, value=Y_true, observed=True)
Y_pred = Normal('Y_pred', mu=A*np.linspace(10, 20, 100)+B, tau=tau)
model = MCMC([Y, Y_pred, A, B, tau])

model.sample(iter=100000,burn=10000,thin=1)

plt.hist(model.trace('A')[:])
plt.hist(model.trace('B')[:])
plt.hist(model.trace('tau')[:])
plt.show()