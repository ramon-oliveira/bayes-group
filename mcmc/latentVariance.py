# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:59:18 2016

@author: amenegola
"""

from pymc import *
import pylab as plt
import numpy as np

heights = [1.7, 1.85, 1.82, 1.75]

mean_heights_population = Normal('mhp', mu = 1.7, tau = 1/3.0)
latent_height_individuals = [Normal('lhi%i'%i, mu = mean_heights_population, tau = 1/0.08) for i in range(len(heights))]      
measured_heights = [Uniform('mh', lower = latent_height_individuals[i]-0.05, upper = latent_height_individuals[i]+0.05, value = h, observed = True) for i, h in enumerate(heights)]

model = MCMC([mean_heights_population] + latent_height_individuals + measured_heights)

model.sample(iter=100000,burn=10000,thin=1)

plt.hist(model.trace('mhp')[:])
plt.show()

plt.hist(model.trace('lhi0')[:])
plt.hist(model.trace('lhi1')[:])
plt.hist(model.trace('lhi2')[:])
plt.hist(model.trace('lhi3')[:])
plt.show()

#%%
model2 = MCMC([mean_heights_population])

model2.sample(iter=100000,burn=10000,thin=1)

plt.hist(model2.trace('mhp')[:])
plt.show()

#%%
uh = Normal('uh', mu = 1.7, tau = 1./9.)
um = Normal('um', mu = 1.7, tau = 1./9.)
ah = Normal('ah', mu = uh, tau = 156.25, value = [1.7, 1.85, 1.82, 1.75], observed = True)
am = Normal('am', mu = um, tau = 156.25, value = [1.6,1.7,1.65], observed = True)
muPop = Normal('muP', mu = 1.7, tau = 1./9.)
precision = InverseGamma('p', alpha = 0.1, beta = 0.1)
a = Normal('a', mu = muPop, tau = precision, value = [1.7, 1.85, 1.82, 1.75], observed = True)
model = MCMC([uh,ah,am,um,muPop,precision,a])
model.sample(iter=100000,burn=10000,thin=1)
plt.hist(model.trace('uh')[:], bins = 100)
plt.show()
#print np.mean(model.trace('uh')[:]>model.trace('um')[:])

plt.hist(model.trace('muP')[:], bins = 100)
plt.show()

plt.hist(model.trace('p')[:], bins = 100)
plt.show()