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
wnorm = (ww - pm.w0)/pm.wa
for i in range(len(ww)):
    phy[i] = np.real(func.voigt(wnorm[i], 1e-9))
phy = phy/ integ.simps(phy)

# Initialaice the intensities vectors to solve the ETR
# Computed as a tensor in zz, ww, mus
II = np.empty((len(zz), len(ww), len(mus)))*0
II = np.repeat(np.repeat(func.plank_wien(ww, pm.T)[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)

plank_Ishape = II*1
mu_shape = np.repeat(np.repeat(mus[np.newaxis,:], len(ww), axis=0)[np.newaxis, :, :], len(zz), axis=0)
phy_shape = np.repeat(np.repeat(phy[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
ww_shape = np.repeat(np.repeat(ww[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)

QQ = II*0
II_new = II*1
QQ_new = QQ*1

# Compute the source function as a tensor in of zz, ww, mus
S00 = pm.eps*plank_Ishape
SLI = S00*1
SI = phy_shape/(phy_shape + pm.r) * SLI + (pm.r/(phy_shape + pm.r)) * plank_Ishape
SQ = SI*0                                           # SQ = 0 (size of SI)

plt.plot(ww, II[0,:,int(0)], color='k', label='Plank intensity')
plt.plot(ww, SI[0,:,0], color='b', label='intensity with line')
plt.plot(ww, SLI[0,:,0], color='r', label='line source function')
plt.legend()
plt.show()
plt.plot(ww, (phy/(phy + pm.r)), color='r', label='line factor')
plt.plot(ww, pm.r/(phy + pm.r), color='b', label='plank factor')
plt.plot(ww, phy_shape[0,:,0], color='k', label='shape of line')
plt.legend()
plt.show()

#  ------------------- FUNCTIONS FOR THE SOLVE METHOD -------------------------------
# Function to compute the coeficients of the Short Characteristics method
def psi_calc(deltaum, deltaup, mode='quad'):

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


# ------------- TEST ON THE SHORT CHARACTERISTICS METHOD ----------------------------
# II = II*0
# II[0] = plank_Ishape[0]
SI = 0*II
SQ = 0*II

for j in range(len(mus)):
    taus = np.exp(-zz)/mus[j]
    deltau = abs(taus[1:] - taus[:-1])
    for i in range(len(zz)-2):

        psim,psio,psip = psi_calc(deltau[i], deltau[i+1])

        # print(i,j, II.shape, QQ.shape, SI.shape, SQ.shape, deltau.shape)
        II_new[i+1,:,j] = II_new[i+1,:,j]*np.exp(-deltau[i]) + SI[i,:,j]*psim + SI[i+1,:,j]*psio + SI[i+2,:,j]*psip
        QQ_new[i+1,:,j] = QQ_new[i+1,:,j]*np.exp(-deltau[i]) + SQ[i,:,j]*psim + SQ[i+1,:,j]*psio + SQ[i+2,:,j]*psip

    psim, psio = psi_calc(deltau[-2], deltau[-1], mode='linear')

    # print(i,j, II.shape, QQ.shape, SI.shape, SQ.shape, deltau.shape)
    II_new[-1,:,j] = II_new[-1,:,j]*np.exp(-deltau[-1]) + SI[-2,:,j]*psim + SI[-1,:,j]*psio 
    QQ_new[-1,:,j] = QQ_new[-1,:,j]*np.exp(-deltau[-1]) + SQ[-2,:,j]*psim + SQ[-1,:,j]*psio

plt.plot(ww, II[0,:,int(pm.qnd/2)])
plt.plot(ww, II_new[99,:,int(pm.qnd/2)])
plt.plot(ww, II_new[75,:,int(pm.qnd/2)])
plt.plot(ww, II_new[90,:,int(pm.qnd/2)])
plt.plot(ww, II_new[80,:,int(pm.qnd/2)])
plt.show()
# print(II-II_new)
plt.plot(zz,II[:,0,0])
plt.plot(zz,II_new[:,0,0])
plt.show()
exit()

# -----------------------------------------------------------------------------------
# ---------------------- MAIN LOOP TO OBTAIN THE SOLUTION ---------------------------
# -----------------------------------------------------------------------------------

w2jujl = (-1)**(1+pm.ju+pm.jl) * np.sqrt(3*(2*pm.ju + 1)) * jsymbols.j3(1, 1, 2, pm.ju, pm.ju, pm.jl)

# while max(diff) > pm.tolerance:
for i in tqdm(range(10)):
    # ----------------- SOLVE RTE BY THE SHORT CHARACTERISTICS ---------------------------
    # print('Solving the Radiative Transpor Equations')
    for j in range(len(mus)):
        taus = np.exp(-zz)/mus[j]
        deltau = abs(taus[1:] - taus[:-1])
        for i in range(len(zz)-2):

            psim,psio,psip = psi_calc(deltau[i], deltau[i+1])

            # print(i,j, II.shape, QQ.shape, SI.shape, SQ.shape, deltau.shape)
            II_new[i+1,:,j] = II_new[i+1,:,j]*np.exp(-deltau[i]) + SI[i,:,j]*psim + SI[i+1,:,j]*psio + SI[i+2,:,j]*psip
            QQ_new[i+1,:,j] = QQ_new[i+1,:,j]*np.exp(-deltau[i]) + SQ[i,:,j]*psim + SQ[i+1,:,j]*psio + SQ[i+2,:,j]*psip

        psim, psio = psi_calc(deltau[-2], deltau[-1], mode='linear')

        # print(i,j, II.shape, QQ.shape, SI.shape, SQ.shape, deltau.shape)
        II_new[-1,:,j] = II_new[-1,:,j]*np.exp(-deltau[-1]) + SI[-2,:,j]*psim + SI[-1,:,j]*psio 
        QQ_new[-1,:,j] = QQ_new[-1,:,j]*np.exp(-deltau[-1]) + SQ[-2,:,j]*psim + SQ[-1,:,j]*psio

    # ---------------- COMPUTE THE COMPONENTS OF THE RADIATIVE TENSOR ----------------------
    # print('computing the components of the radiative tensor')

    Jm00 = 1/2 * integ.simps(integ.simps(phy_shape*II_new))
    Jm02 = 1/np.sqrt(4**2 * 2) * integ.simps(
                                 integ.simps(
                                     phy_shape * (3*mu_shape**2 - 1)*II_new + 3*(mu_shape**2 - 1)*QQ_new ))

    # print('The Jm00 component is: {} and the Jm02 is: {}'.format(Jm00[-1],Jm02[-1]))
    # ---------------- COMPUTE THE SOURCE FUNCTIONS TO SOLVE THE RTE -----------------------
    # print('Computing the source function to close the loop and solve the ETR again')

    Jm00_shape = np.repeat(np.repeat(Jm00[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)
    Jm02_shape = np.repeat(np.repeat(Jm02[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)

    S00 = (1-pm.eps)*Jm00_shape + pm.eps*plank_Ishape
    S20 = pm.Hd * (1-pm.eps)/(1 + (1-pm.eps)*pm.dep_col) * w2jujl**2 * Jm02_shape

    SLI = S00 + w2jujl * (3*mu_shape**2 - 1)/np.sqrt(8) * S20
    SLQ = w2jujl * 3*(mu_shape**2 - 1)/np.sqrt(8) * S20

    SI = phy_shape/(phy_shape + pm.r)*SLI + pm.r/(phy_shape + pm.r)*plank_Ishape
    SQ = phy_shape/(phy_shape + pm.r)*SLQ

    diff = II - II_new

    II = II_new
    QQ = QQ_new

plt.plot(ww, II[0,:,int(pm.qnd/2)])
plt.plot(ww, SI[40,:,0])
plt.plot(ww, SLI[40,:,0])
plt.show()
plt.plot(zz, SI[:,0,0])
plt.show()

# tolerancia en todas las capas en SO0, S02

# -------------------- ONCE WE OBTAIN THE SOLUTION, COMPUTE THE POPULATIONS ----------------
Jm00_shape = np.repeat(np.repeat(Jm00[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)
Jm02_shape = np.repeat(np.repeat(Jm02[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)

rho00 = np.sqrt((2*pm.ju + 1)/(2*pm.jl+1)) * (2*cte.h*ww_shape**3/cte.c**2)**-1 * \
    ((1-pm.eps)*Jm00_shape + pm.eps*plank_Ishape)/((1-pm.eps)*pm.dep_col + 1)

rho02 = np.sqrt((2*pm.ju + 1)/(2*pm.jl+1)) * (2*cte.h*ww_shape**3/cte.c**2)**-1 * \
    ((1-pm.eps)*Jm02_shape)/((1-pm.eps)*(1j*pm.Hd*2 + pm.dep_col) + 1)
