import numpy as np
from math import ceil
from matplotlib import pyplot as plt

""" Parameters
S = susceptible to antibiotic
P = produces and is resistant to antibiotic, can transfer its plasmid to either S or R
R = resistant to antibiotic, can transfer its plasmid to S
"""

rS = 1 # growth rate of wild-type
rP = 0.85 # growth rate of P
rR = 0.9 # growth rate of R
d = 0.5 # death rate of all species
lambdaP = 0.001 # horizontal transfer of P plasmid
lambdaR = 0.001 # horizontal transfer of R plasmid
Pcr = 0.05 # concentration of P for which death rate of W doubles
kappa = 1 # cooperative binding of antibiotic
f = lambda P: rS * P**kappa/(Pcr**kappa + P**kappa) # decrease in growth rate (or increase in death rate) of S from P producing antibiotic

Sdot = lambda S, P, R: (rS - (d + f(P)) - P * lambdaP - R * lambdaR) * S
Pdot = lambda S, P, R: (rP - d + S * lambdaP + R * lambdaP) * P
Rdot = lambda S, P, R: (rR - d + S * lambdaR - P * lambdaP) * R

dt = 0.01
tmax = 5000
steps = ceil(tmax/dt)

time = np.zeros(steps)
S = np.zeros(steps)
P = np.zeros(steps)
R = np.zeros(steps)

time[0] = 0
P[0] = 0.01 # how much P do we inject as a proportion of total cells
R[0] = 1e-4 # what fraction of the population is already resistant?
S[0] = 1 - P[0] - R[0]

for t in range(1, steps):
    time[t] = time[t - 1] + dt
    
    dS = Sdot(S[t - 1], P[t - 1], R[t - 1]) * dt
    dP = Pdot(S[t - 1], P[t - 1], R[t - 1]) * dt
    dR = Rdot(S[t - 1], P[t - 1], R[t - 1]) * dt
    dphi = dS + dP + dR
    
    # subtract average fitness
    dS -= dphi * S[t - 1]
    dP -= dphi * P[t - 1]
    dR -= dphi * R[t - 1]
    
    S[t] = S[t - 1] + dS
    P[t] = P[t - 1] + dP
    R[t] = R[t - 1] + dR
    
plt.plot(time, S, 'b', time, P, 'g', time, R, 'r')
plt.title('Strain abundances')
plt.legend(['Susceptible', 'Producing', 'Resistant'])
plt.show()

plt.plot(time, P/R)
plt.title('Producing vs Resistant')
plt.show()
