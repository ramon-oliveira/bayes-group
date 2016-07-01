# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 00:01:14 2016

@author: tabacof
"""

import pystan
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
from brasileirao2015 import brasileirao

cols = ['data', 'hora', 'time_casa', 'time_fora', 'gols_casa', 'gols_fora']
df = pd.DataFrame(columns=cols)
for jogo in brasileirao:
    gols_casa = jogo[10].split(' : ')[0]
    gols_fora = jogo[10].split(' : ')[1]
    df.loc[len(df)] = [jogo[2], jogo[3], jogo[5], jogo[8], gols_casa, gols_fora]
print('Jogos ganhos em casa:', len(df[df.gols_casa > df.gols_fora]))
print('Jogos ganhos fora de casa:', len(df[df.gols_casa < df.gols_fora]))

set(df.time_casa) | set(df.time_fora)

team_idx = {team: i for i, team in enumerate(set(df.time_casa) | set(df.time_fora), start=1)}
idx_team = {idx: team for team, idx in team_idx.items()}

matches = []
for _, row in df.iterrows():
    time_casa = row[2]
    time_fora = row[3]
    gols_casa = row[4]
    gols_fora = row[5]
    matches.append([team_idx[time_casa], team_idx[time_fora], gols_casa, gols_fora])
matches = np.array(matches, dtype=np.int64)

code = '''
data {
    int n_matches;
    int n_teams;
    int matches[n_matches, 4]; // time_casa, time_fora, gols_casa, gols_fora
}
parameters {
    real C; // casa-fora
    real B; // bias
    real A[n_teams]; // ataque
    real D[n_teams]; // defesa
}
model {
    C ~ normal(0, 1); 
    B ~ normal(0, 1); 
    A ~ normal(0, 1); 
    D ~ normal(0, 1); 
    
    for(i in 1:n_matches){
        matches[i, 3] ~ poisson(exp(C + A[matches[i, 1]] - D[matches[i, 2]] + B));
        matches[i, 4] ~ poisson(exp(A[matches[i, 2]] - D[matches[i, 1]] + B));
    }
}
'''
    
data = {
    'n_matches': len(matches),
    'n_teams': len(set(matches[:,0]) | set(matches[:,1])),
    'matches': matches,
}

fit = pystan.stan(model_code=code, data=data, iter=1000, chains=4)

print(fit)
fit.plot()
trace = fit.extract()

score = np.mean(trace['A'], axis = 0)+np.mean(trace['D'], axis = 0)
sorted_scores = sorted(zip(idx_team,score), key=lambda tup: tup[1])
for s in sorted_scores:
    print(idx_team[s[0]], s[1])
