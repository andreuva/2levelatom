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
from jsymbols import jsymbols
jsymbols = jsymbols()

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
II = np.empty((len(zz), len(ww), len(mus)))*0
II = np.repeat(np.repeat(func.plank_nu(ww, pm.T)[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)

plank_Ishape = II
mu_shape = np.repeat(np.repeat(mus[np.newaxis,:], len(ww), axis=0)[np.newaxis, :, :], len(zz), axis=0)
phy_shape = np.repeat(np.repeat(phy[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)

QQ = II*0
II_new = II
QQ_new = QQ

# Compute the source function as a matrix of zz and ww
SI = np.repeat(np.repeat(pm.r/(phy + pm.r)[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0) * II
SQ = SI*0                                           # SQ = 0 (size of SI)

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
for i in range(10):
    print('Solving the Radiative Transpor Equations')
    for j in tqdm(range(len(mus))):
        ss = zz/mus[j]
        ChiI = 1
        taus = -ss*ChiI
        deltau = taus[1:] - taus[:-1]

        for i in range(len(zz)-2):

            psim,psio,psip = psi_calc(deltau[i],deltau[i+1])

            # print(i,j, II.shape, QQ.shape, SI.shape, SQ.shape, deltau.shape)
            II_new[i+1,:,j] = II[i+1,:,j] + SI[i,:,j]*psim + SI[i+1,:,j]*psio + SI[i+2,:,j]*psip
            QQ_new[i+1,:,j] = QQ[i+1,:,j] + SQ[i,:,j]*psim + SQ[i+1,:,j]*psio + SQ[i+2,:,j]*psip

    # ---------------- COMPUTE THE COMPONENTS OF THE RADIATIVE TENSOR ----------------------
    print('computing the components of the radiative tensor')

    Jm00 = 1/2 * integ.simps( phy * integ.simps(II_new) )
    Jm02 = 1/np.sqrt(4**2 * 2) * integ.simps( phy * integ.simps( (3*mus**2 - 1)*II_new + 3*(mus**2 - 1)*QQ_new ))

    print('The Jm00 component is: {} and the Jm02 is: {}'.format(Jm00[-1],Jm02[-1]))
    # ---------------- COMPUTE THE SOURCE FUNCTIONS TO SOLVE THE RTE -----------------------
    print('Computing the source function to close the loop and solve the ETR again')

    w2jujl = (-1)**(1+pm.ju+pm.jl) * np.sqrt(3*(2*pm.ju + 1)) * jsymbols.j3(1, 1, 2, pm.ju, pm.ju, pm.jl)

    S00 = (1-pm.eps)*np.repeat(np.repeat(Jm00[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2) + pm.eps*plank_Ishape
    S20 = pm.Hd * (1-pm.eps)/(1 + (1-pm.eps)*pm.dep_col) * w2jujl**2 * np.repeat(np.repeat(Jm02[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)

    SLI = S00 + w2jujl * (3*mu_shape**2 - 1)/np.sqrt(8) * S20
    SLQ = w2jujl * 3*(mu_shape**2 - 1)/np.sqrt(8) * S20

    SI = phy_shape/(phy_shape + pm.r)*SLI + pm.r/(phy_shape + pm.r)*plank_Ishape
    SQ = phy_shape/(phy_shape + pm.r)*SLQ

    II = II_new
    QQ = QQ_new


# -------------------- ONCE WE OBTAIN THE SOLUTION, COMPUTE THE POPULATIONS ----------------
rho00 = np.sqrt((2*pm.ju + 1)/(2*pm.jl+1)) * (2*cte.h*ww**3/cte.c**2)**-1 * \
    ((1-pm.eps)*Jm00 + pm.eps*plank_Ishape)/((1-pm.eps)*pm.dep_col)