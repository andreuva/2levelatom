#############################################################################
#                2 LEVEL ATOM ATMOSPHERE SOLVER                             #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ
# import to make progress bars in the loops
from tqdm import tqdm
# local imports of constants parameters and functions
import constants as cte
import parameters as pm
import physical_functions as func

# We define the z0, zl, dz as our heigt grid (just 1D because of a
# plane-parallel atmosfere and axial-simetry)
zz = np.arange(pm.zl, pm.zu, pm.dz)          # compute the 1D grid

# Define the grid in frequencies ( or wavelengths )
ww = np.arange(pm.wl, pm.wu, pm.dw)          # Compute the 1D spectral grid

# Define the directions of the rays
mus = np.linspace(0.01, 1, pm.qnd)

# ------------------------ INITIAL CONDITIONS -----------------------------
# Compute the initial Voigts vectors
phy = np.empty_like(ww)
for i in range(len(ww)):
    phy[i] = np.real(func.voigt(ww[i], pm.a))

# Initialaice the intensities vectors to solve the ETR
II = np.empty((len(zz), len(ww), len(mus)))
II[0] = np.repeat(func.plank_nu(ww, pm.T)[ :, np.newaxis], len(mus), axis=1)
QQ = II*0
II_new = II
QQ_new = QQ

# Compute the source function as a matrix of zz and ww
SI = pm.r/(phy + pm.r)*func.plank_nu(ww, pm.T)
SI = np.repeat( SI[ :, np.newaxis], len(mus), axis=1)
SQ = SI*0                                           # SQ = 0 (size of SI)
print(SI.shape)

# ------------------- FUNCTIONS FOR THE SOLVE METHOD -------------------------------
# Function to compute the coeficients of the Short Characteristics method
def psi_calc(deltaum, deltaup, mode = 'quad'):

    U0 = 1 - np.exp(-deltaum)
    U1 = deltaum*np.exp(-deltaum) - U0
    U2 = deltaum**2*np.exp(-deltaum) - 2*U1
    
    if mode == 'quad':
        psim = U0 + (U2 - U1*(deltaup + 2*deltaum))/(deltaum*(deltaum + deltaup))
        psio = (U1*(deltaum + deltaup) - U2)/(deltaum*deltaup)
        psip = (U2 - U1*deltaum)/(deltaup*(deltaup+deltaum))
        return psim, psio, psip
    elif mode == 'linear':
        psim = U0*deltaum - U1/deltaum
        psio = U1/deltaum
        return psim, psio
    else:
        raise Exception('mode should be quad or lineal but {} was introduced'.format(mode))

# ----------------- SOLVE RTE BY THE SHORT CHARACTERISTICS ---------------------------
print('Solving the Radiative Transpor Equations')
for j in tqdm(range(len(mus))):
    ss = zz/mus[j]
    ChiI = 1
    taus = -ss*ChiI
    deltau = taus[1:] - taus[:-1]

    for i in range(len(zz)-2):

        psim,psio,psip = psi_calc(deltau[i],deltau[i+1])

        # print(i,j, II.shape, QQ.shape, SI.shape, SQ.shape, deltau.shape)
        II_new[i+1,:,j] = II[i,:,j] + SI[:,j]*psim + SI[:,j]*psio + SI[:,j]*psip
        QQ_new[i+1,:,j] = QQ[i,:,j] + SQ[:,j]*psim + SQ[:,j]*psio + SQ[:,j]*psip

# ---------------- COMPUTE THE COMPONENTS OF THE RADIATIVE TENSOR ----------------------
print('computing the components of the radiative tensor')

Jm00 = 1/2 * integ.simps( phy * integ.simps(II_new) )
Jm02 = 1/np.sqrt(4**2 * 2) * integ.simps( phy * integ.simps( (3*mus**2 - 1)*II_new + 3*(mus**2 - 1)*QQ_new ))

print(Jm00.shape,Jm02.shape)

print('The Jm00 component is: {} and the Jm02 is: {}'.format(Jm00,Jm02))
