import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
import sys
import os
pi = np.pi
nb_stdout = sys.stdout

def sin(x):
    return np.sin(x)
def cos(x):
    return np.cos(x)
def tan(x):
    return np.tan(x)

# Disable Printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore Printing
def enablePrint():
    sys.stdout = nb_stdout



## Counts the number of things in each pixel and maps it 
## to the pixel ordered array
def pixarr(nside,theta,phi):
    val = hp.ang2pix(nside,theta,phi)
    npix = hp.nside2npix(nside)
    out = np.zeros(npix)
    
    for i in range(npix):
        ok = np.where(val == i)
        if np.size(ok) == 0:
            out[i] = hp.UNSEEN
        else:    
            out[i] = np.size(ok)
        
    return out

## Averages the residual for each pixel
def pixres(nside,theta,phi,res):
    val = hp.ang2pix(nside,theta,phi)
    npix = hp.nside2npix(nside)
    out = np.zeros(npix)
    
    for i in range(npix):
        ok = np.where(val == i)
        if np.size(ok) == 0:
            out[i] = hp.UNSEEN
        else:
            out[i] = np.mean(res[ok])
        
    return out

def pixresm(nside,theta,phi,res,z):
    val = hp.ang2pix(nside,theta,phi)
    npix = hp.nside2npix(nside)
    out = np.zeros(npix)
    out1 = np.copy(out)
    out2 = np.copy(out)
    
    for i in range(npix):
        ok = np.where(val == i)
        if np.size(ok) == 0:
            out[i] = hp.UNSEEN
        else:
            out[i] = np.mean(res[ok])
        out1[i] = len(res[ok])
        out2[i] = np.mean(z[ok])
    return out, out1, out2

def ranres(num, me, sd):
    out = np.zeros(len(num))
    for i in range(len(num)):
        if num[i] == 0.:
            out[i] = hp.UNSEEN
        else:
            out[i] = np.random.normal(me,sd/np.sqrt(num[i]))
            
    return out

def CIL(C,a,b):
    out = np.zeros([np.max(a)+1,4])
    for i in range(np.max(a)+1):
        lis = sorted(b[(a == i)])
        mid = np.median(lis)
        n = int(np.where(lis == mid)[0])
        top = lis[int(n+C/2*len(lis))]
        bot = lis[int(n-C/2*len(lis))]
        boter = mid-bot
        toper = top-mid
        out[i,0] = i
        out[i,1] = mid
        out[i,2] = boter
        out[i,3] = toper
        
    return out

def powspec(nside, theta, phi, res, Conf = 0.68, remove_monopole = True, remove = [], shift=False):
    #Generating a map of the residuals
    residuals = pixres(nside,theta,phi,res)
    numresiduals = pixarr(nside,theta,phi)
    
    residuals[remove] = hp.UNSEEN
    numresiduals[remove] = 0.
    for i in range(len(numresiduals)):
        if numresiduals[i] == hp.UNSEEN:
            numresiduals[i] = 0.
            
    
    print 'Number of SN Used:', sum(numresiduals)
    blockPrint()
    if remove_monopole == True:
        residuals = hp.remove_monopole(residuals)
    enablePrint()
    
    #Finding Norm Factor
    dA = hp.nside2pixarea(nside)
    omg_sky = 4*pi - dA*np.count_nonzero(residuals == hp.UNSEEN)
    norz = (4*pi/omg_sky)
    
    #Generating Data Spectrum
    cl_dat = hp.anafast(residuals)
    ell_dat = np.arange(len(cl_dat))
    
    #Running the Simulations and Finding Normal Spectrum + Error Bars
    x = []
    y = []

    mres = np.copy(res)
    for i in range(1001):
        np.random.shuffle(mres)
        mresiduals = pixres(nside, theta, phi, mres)
        
        blockPrint()
        if remove_monopole == True:
            mresiduals = hp.remove_monopole(mresiduals)
        enablePrint()
        
        cls = hp.anafast(mresiduals)
        ell = np.arange(len(cls))
    
        x.extend(ell)
        y.extend(cls)
    
    x = np.array(x)
    y = np.array(y)
    
    CI = CIL(Conf, x, y)
    
    ell_sim = CI[:,0]
    cl_sim = CI[:,1]
    err_sim = [CI[:,2],CI[:,3]]
    print "done"
    return ell_dat, cl_dat, ell_sim, cl_sim, err_sim, norz

def CIres(cls, err, cld):
    out = np.zeros(len(cls))
    upbound = cls + err[1]
    lowbound = cls - err[0]
    for i in range(len(cls)):
        if cld[i] < upbound[i] and cld[i] > lowbound[i]:
            out[i] = True
        else:
            out[i] = False
            
    return out

def chi_sq(dat, mid, err):
    out = 0.
    for i in range(len(dat)):
        sigma = (err[0,i]+err[1,i])/2
        out += ((mid[i] - dat[i])/sigma)**2
    return out

def shift_cov(keep_indicies, cov):
    output = np.zeros((np.size(keep_indicies), np.size(keep_indicies)))
    
    
    for i in range(np.size(keep_indicies)):
        for j in range(np.size(keep_indicies)):
            output[i][j] = cov[keep_indicies[i]][keep_indicies[j]]
            
    return output
            
                