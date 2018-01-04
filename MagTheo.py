import math as math
import numpy as np
from numpy.linalg import det, inv
import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow, show, gray, colorbar, ylabel, xlabel, title
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import csv

#Finding m_th from H0dL
def m_theory(d):
    m_th = np.zeros(len(d))
    for k in range(len(d)):
        m_th[k] = 5*math.log(d[k],10)
    return m_th

#matching redshifts to a tabulation
def match(x, y, episilon):
    tab_val = np.zeros(len(y))
    for i in range(len(y)):
        for j in range(len(x)):
            err = abs(x[j] - y[i])
            if err <= episilon:
                tab_val[i] = int(j)
    return tab_val

#Updated chi function, note that the the chi sum is now split into 2 parts
def chi(x, y, z, C_inv):
    #part 1 of chi sum: Matrix multiplication (delm^T * C^-1 * delm)
    low_z_lim = len(C_inv)
    delta_mag = y - x
    A = np.dot(np.transpose(delta_mag[0:low_z_lim]), C_inv)
    X_sq = np.dot(A, delta_mag[0:low_z_lim])
    
    #part 2 of chi sum: Standard (sum delm^2 / sig^2)
    for i in range(low_z_lim, len(x)):
        X_sq += delta_mag[i]**2 / z[i]**2
    
    return X_sq

def chi2(x, y, z):
    for i in range(low_z_lim, len(x)):
        X_sq += delta_mag[i]**2 / z[i]**2
    
    return X_sq


#Finding H0dL given theory and redshift
def d_L(z, omega_m, tabulation, x):
    H0dL = np.zeros(len(z))
    d_L_tab = np.zeros(len(x))
    dx = max(x)/len(x)
    omega_de = 1 - omega_m
    
    
    for i in range(len(x)):
        if i != 0:
            d_L_tab[i] = dx / np.sqrt(omega_m * (1 + x[i])**3 + omega_de)                                    + d_L_tab[i-1]
            
        else:
            d_L_tab[i] = dx / np.sqrt(omega_m * (1 + x[i])**3 + omega_de)
            
    for j in range(len(z)):
        H0dL[j] = (1 + z[j]) * d_L_tab[int(tabulation[j])]
    
    return H0dL

#Likelihood calculation
def likelihood(chi_sq, adj):
    L = np.exp((-chi_sq / 2) + adj)
    return L

#Levels
def lev(A):
    
    Li = np.sort(-1.* A, axis=None)
    Li = -1* Li
    total = sum(Li)
    
    end = [0.68,0.95,0.997]
    out = [0.]
    
    for j in range(0,3,1):
        i = 0
        A = 0.
        while A < total*end[2-j]:
            A += Li[i]   
            i += 1
        out.append(Li[i])
    out.append(max(Li))
    return out;

#Printing a CI graph, returns the mean of a 2D likelihood array
def graphnmean(L_hood, x, y, xlabel, ylabel):
    #Normalize
    graph = L_hood / np.max(L_hood)
    
    #Simple set up of axis and intervals
    intervals = lev(graph)
    Xax, Yax = np.meshgrid(x, y)
    
    #CI and the mean 
    interest = np.where(graph == 1.0)
    meanx = float(x[interest[1].ravel()])
    meany = float(y[interest[0].ravel()])
    
    inner = np.where(graph >= intervals[3])
    rx = inner[1].ravel()
    ry = inner[0].ravel()
    rxint = []
    ryint = []
    for i in range(len(rx)):
        rxint.append(float(x[rx[i]]))
    for i in range(len(ry)):
        ryint.append(float(y[ry[i]]))
    
    lowx = float(min(rxint))
    highx = float(max(rxint))
    
    lowy = float(min(ryint))
    highy = float(max(ryint))
    
    print xlabel, " Mean:"
    print meanx, '(+', highx - meanx, ', -', meanx - lowx, ')'
    
    print ylabel, " Mean:"
    print meany, '(+', highy - meany, ', -', meany - lowy, ')'
    
    #Plotting
    plt.contourf(Xax, Yax, graph, levels = intervals, colors = ['white', 'midnightblue', 'limegreen', 'red'])
    plt.plot(meanx, meany, '*', color='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Likelihood Varying " + xlabel + " and " + ylabel)
    
    RED = mpatches.Patch(color='red', label = '68% Confidence')
    GREEN = mpatches.Patch(color='limegreen', label = '95% Confidence')
    BLUE = mpatches.Patch(color='midnightblue', label = '99.7% Conifence')
    Average = mlines.Line2D([], [], color='black', marker='*',
                      markersize=15, label='Mean')

    plt.legend(handles=[RED, GREEN, BLUE, Average], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()
    
    return;