import numpy as np
from math import ceil
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Differential Equation Simulator", epilog = "")

parser.add_argument('-p', '--plot', help="Plot the default trajectory", action="store_true")
parser.add_argument('-c', '--concentration', 
                    help="Vary the concentrations of P and R and plot success", action="store_true")
parser.add_argument('-l', '--legend', help="Display a color legend", action="store_true")

args = parser.parse_args()



""" Parameters
S = susceptible to phage
I = infected with phage, lysogenic state
L = infected with phage, lytic state
P = phage
R = resistant to phage
"""

def simulate(r=2, KS = 1e8, KI = 7e7, KR = 9e7, alpha = 1e-9, c = 1, lambdaP = 0.15, b = 100, bRP = 90, gamma = 0.1, P0=1e9, L0 = 1, S0 = 1e6, R0=1, tmax=15000, getImax=False):
    """ This method uses Euler's method to simulate the differential equations involved.
    
        r = growth rate
        K* = carrying capacity of *
        alpha = reaction rate of phage and cells
        c = conversion rate of infected to lytic cells
        lambdaP = lysis rate
        b = phage burst size
        gamma = death rate of phage
        P0 = initial concentration of P
        S0 = initial concentration of S
        R0 = initial concentration of R
        *initial concentrations of I and L are assumed 0*
        tmax = total simulation time
        getImax = get first maximum value of I
    """
    f = lambda P: rS * P**kappa/(Pcr**kappa + P**kappa) # decrease in growth rate (or increase in death rate) of S from P producing antibiotic

    Sdot = lambda S, I, L, R, P: (r * (1 - (S + I + R)/KS) - alpha * P) * S # add L back
    #Idot = lambda S, I, L, R, P: (r * (1 - (S + I + L + R)/KI) - c) * I + alpha * S * P
    #Ldot = lambda S, I, L, R, P: c * I - lambdaP * L
    #Pdot = lambda S, I, L, R, P: b * lambdaP * L - gamma * P
    Idot = lambda S, I, L, R, P: 0
    Ldot = lambda S, I, L, R, P: bRP * alpha * L * R - gamma * L
    Pdot = lambda S, I, L, R, P: b * alpha * P * S - gamma * P
    Rdot = lambda S, I, L, R, P: (r * (1 - (S + I + R)/KR) - alpha * L) * R # add L back

    dt = 0.01
    steps = ceil(tmax/dt)

    time = np.zeros(steps)
    S = np.zeros(steps)
    I = np.zeros(steps)
    L = np.zeros(steps)
    R = np.zeros(steps)
    P = np.zeros(steps)

    time[0] = 0
    R[0] = R0 # how many cells are already already resistant?
    S[0] = S0 # how many cells does the simulation begin with?
    I[0] = 0 # simulation starts with no infected cells
    L[0] = L0
    P[0] = P0 # how many phage do we inject at the beginning

    for t in range(1, steps):
        time[t] = time[t - 1] + dt
        
        dS = Sdot(S[t - 1], I[t - 1], L[t - 1], R[t - 1], P[t - 1]) * dt
        dI = Idot(S[t - 1], I[t - 1], L[t - 1], R[t - 1], P[t - 1]) * dt
        dL = Ldot(S[t - 1], I[t - 1], L[t - 1], R[t - 1], P[t - 1]) * dt
        dR = Rdot(S[t - 1], I[t - 1], L[t - 1], R[t - 1], P[t - 1]) * dt
        dP = Pdot(S[t - 1], I[t - 1], L[t - 1], R[t - 1], P[t - 1]) * dt
        
        S[t] = S[t - 1] + dS
        I[t] = I[t - 1] + dI
        L[t] = L[t - 1] + dL
        R[t] = R[t - 1] + dR
        P[t] = P[t - 1] + dP
        
        # we're getting the first maximum of I
        # this happens whenever the value of I decreases
        if getImax and I[t] < I[t-1]:
            return I[t-1]
    
    if getImax:
        return max(I)
    return (time, S, I, L, R, P)
    
def plotResults(time, S, I, L, R, P):
    """ Given time series for the strains, plots them all """
    plt.plot(time, S, time, I, time, L, time, R, time, P)
    plt.yscale('log')
    plt.title('Fractional abundances of strains')
    plt.legend(['Susceptible', 'Infected', 'Lytic', 'Resistant', 'Phage'])
    plt.ylim([1, 1e12])
    plt.xlabel('Time (hours)')
    plt.ylabel('Strain abundances')
    plt.show()
    
'''    plt.plot(time, S, 'b', time, P, 'g', time, R, 'r')
    plt.title('Fractional abundances of strains')
    plt.legend(['Susceptible', 'Producing', 'Resistant'])
    plt.ylim([-0.01, 1.01])
    plt.xlim([0, 300])
    plt.xlabel('Time (hours)')
    plt.ylabel('Strain abundances')
    plt.show()'''
    
def main():
    if args.plot:
        (time, S, I, L, P, R) = simulate()
        print(max(I))
        plotResults(time, S, I, L, P, R)
    if args.legend:
        colorLegend(0.85,0.002)
    if args.concentration:
        Pmax = []
        Pvals = []
        Rvals = []
        for P_0 in np.arange(-4,0,0.1):
            for R_0 in np.arange(-6,-1.9,0.1):
                Pvals.append(P_0)
                Rvals.append(R_0)
                Pmax.append(simulate(P0=10**P_0, R0=10**R_0, getPmax=True))
                
        Pvals = np.array(Pvals)
        Rvals = np.array(Rvals)
        Pmax = np.array(Pmax)
        
        g = np.array([0, 1, 0])
        r = np.array([1, 0, 0])
        
        corr_r = 0.005/(1.005 - Pmax) # correction to the color; is 1 if Pmax = 1, and is small for Pmax small
        corr_g = 0.05/(1.05 - Pmax)
        plt.scatter(np.array(Pvals), np.array(Rvals), c=g[np.newaxis,:]*corr_g[:, np.newaxis]
                    +r[np.newaxis,:]*(1-corr_r[:, np.newaxis]),
                    s=100, marker='s', edgecolors='none')
        plt.xlabel('log(Concentration of P)')
        plt.ylabel('log(Concentration of R)')
        plt.title('Effectiveness')
        print(Pmax)
        plt.show()

        
def colorLegend(minX,stepX):
    x = []
    y = []
    for t in np.arange(minX, 1, stepX):
        for s in np.arange(0,1,0.02):
            x.append(t)
            y.append(s)
    x = np.array(x)
    y = np.array(y)
    corr_r = 0.005/(1.005 - x) - 1e-6
    corr_g = 0.05/(1.05 - x) - 1e-6
    g = np.array([0, 1, 0])
    r = np.array([1, 0, 0])
    
    plt.scatter(x, y, c=g[np.newaxis,:]*corr_g[:, np.newaxis]
               +r[np.newaxis,:]*(1-corr_r[:, np.newaxis]),
               s=100, marker='s', edgecolors='none')
    plt.xlabel('Maximum concentration of P')
    ax1 = plt.axes()
    ax1.axes.get_yaxis().set_visible(False)
    plt.xlim([minX,1])
    plt.ylim([0,0.9])
    plt.title('Color Legend')
    plt.show()
     

if __name__ == '__main__':
    main()
