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
S = susceptible to antibiotic
P = produces and is resistant to antibiotic, can transfer its plasmid to either S or R
R = resistant to antibiotic, can transfer its plasmid to S
"""

def simulate(rS=1, rP=0.85, rR=0.9, d=0.5, lambdaP=0.001, lambdaR=0.001, Pcr=0.05, kappa=1, P0=0.01, R0=1e-4, tmax=3000, getPmax=False):
    """ This method uses Euler's method to simulate the differential equations involved.
    
        rS = growth rate of wild-type
        rP = growth rate of P
        rR = growth rate of R
        d = death rate of all species
        lambdaP = horizontal transfer of P plasmid
        lambdaR = horizontal transfer of R plasmid
        Pcr = concentration of P for which death rate of W doubles
        kappa = cooperative binding of antibiotic
        P0 = initial concentration of P
        R0 = initial concentration of R
        tmax = total simulation time
        getPmax = get first maximum value of P
    """
    f = lambda P: rS * P**kappa/(Pcr**kappa + P**kappa) # decrease in growth rate (or increase in death rate) of S from P producing antibiotic

    Sdot = lambda S, P, R: (rS - (d + f(P)) - P * lambdaP - R * lambdaR) * S
    Pdot = lambda S, P, R: (rP - d + S * lambdaP + R * lambdaP) * P
    Rdot = lambda S, P, R: (rR - d + S * lambdaR - P * lambdaP) * R

    dt = 0.01
    steps = ceil(tmax/dt)

    time = np.zeros(steps)
    S = np.zeros(steps)
    P = np.zeros(steps)
    R = np.zeros(steps)

    time[0] = 0
    P[0] = P0 # how much P do we inject as a proportion of total cells
    R[0] = R0 # what fraction of the population is already resistant?
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
        
        # we're getting the first maximum of P
        # this happens whenever the value of P decreases
        if getPmax and P[t] < P[t-1]:
            return P[t-1]
    
    if getPmax:
        return max(P)
    return (time, S, P, R)
    
def plotResults(time, S, P, R):
    """ Given time series for the strains, plots them all """
    plt.plot(time, S, 'b', time, P, 'g', time, R, 'r')
    plt.title('Fractional abundances of strains')
    plt.legend(['Susceptible', 'Producing', 'Resistant'])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Time (hours)')
    plt.ylabel('Strain abundances')
    plt.show()
    
    plt.plot(time, S, 'b', time, P, 'g', time, R, 'r')
    plt.title('Fractional abundances of strains')
    plt.legend(['Susceptible', 'Producing', 'Resistant'])
    plt.ylim([-0.01, 1.01])
    plt.xlim([0, 300])
    plt.xlabel('Time (hours)')
    plt.ylabel('Strain abundances')
    plt.show()
    
def main():
    if args.plot:
        (time, S, P, R) = simulate()
        print(max(P))
        plotResults(time, S, P, R)
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
