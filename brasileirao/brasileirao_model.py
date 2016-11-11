import pystan
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import scipy.stats
import pandas as pd
from brasileirao2015 import brasileirao
import random

cols = ['data', 'hora', 'time_casa', 'time_fora', 'gols_casa', 'gols_fora']
df = pd.DataFrame(columns=cols)
for jogo in brasileirao:
    gols_casa = jogo[10].split(' : ')[0]
    gols_fora = jogo[10].split(' : ')[1]
    df.loc[len(df)] = [jogo[2], jogo[3], jogo[5], jogo[8], gols_casa, gols_fora]

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
    //real C[n_teams]; // casa-fora
    real C; // casa-fora
    //real B[n_teams]; // bias
    real B; // bias
    real A[n_teams]; // ataque
    real D[n_teams]; // defesa
}
transformed parameters {
    real LAMBDA[n_teams]; // poisson models
    for(i in 1:n_matches){
        //LAMBDA[matches[i, 1]] <- exp(C[matches[i, 1]]*1 + A[matches[i, 1]] - D[matches[i, 2]] + B[matches[i, 1]]);
        //LAMBDA[matches[i, 2]] <- exp(C[matches[i, 2]]*0 + A[matches[i, 2]] - D[matches[i, 1]] + B[matches[i, 2]]);
        LAMBDA[matches[i, 1]] <- exp(C*1 + A[matches[i, 1]] - D[matches[i, 2]] + B);
        LAMBDA[matches[i, 2]] <- exp(C*0 + A[matches[i, 2]] - D[matches[i, 1]] + B);
    }
}
model {
    C ~ normal(0, 1);
    B ~ normal(0, 1);
    A ~ normal(0, 1);
    D ~ normal(0, 1);

    for(i in 1:n_matches){
        matches[i, 3] ~ poisson(LAMBDA[matches[i, 1]]);
        matches[i, 4] ~ poisson(LAMBDA[matches[i, 2]]);
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

trace = fit.extract()

teams_to_plot = [10, 11, 13]
ylim = [0, 0.6]
plt.figure(figsize=(12,7))

for i in teams_to_plot:
    plt.subplot(2,2,1)
    plt.title('A - Ataque')
    hist, vals = np.histogram(trace['A'][:, i-1], range=[-2, 2], bins=20)
    plt.plot(np.linspace(-2, 2, len(hist)), hist/hist.sum(), label=idx_team[i])
    plt.ylim(ylim)
    plt.legend()

    plt.subplot(2,2,2)
    plt.title('D - Defesa')
    hist, vals = np.histogram(trace['D'][:, i-1], range=[-2, 2], bins=20)
    plt.plot(np.linspace(-2, 2, len(hist)), hist/hist.sum(), label=idx_team[i])
    plt.ylim(ylim)
    plt.legend()

    plt.subplot(2,2,3)
    plt.title('LAMBDA - GOLS')
    hist, vals = np.histogram(trace['LAMBDA'][:, i-1], range=[0, 3], bins=20)
    plt.plot(np.linspace(0, 3, len(hist)), hist/hist.sum(), label=idx_team[i])
    plt.ylim(ylim)
    plt.legend()

plt.show()