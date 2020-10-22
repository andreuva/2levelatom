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
mus = np.linspace(-1, 1, pm.qnd)

# ------------------------ SOME INITIAL CONDITIONS -----------------------------
# Compute the initial Voigts vectors
phy = np.empty_like(ww)
wnorm = (ww - pm.w0)/pm.wa          # normalice the frequency to compute phy
for i in range(len(ww)):
    phy[i] = np.real(func.voigt(wnorm[i], 1e-9))
phy = phy/integ.simps(phy)          # normalice phy to sum 1

# Initialaice the intensities vectors to solve the ETR
# Computed as a tensor in zz, ww, mus
plank_Ishape = np.repeat(np.repeat(func.plank_wien(ww, pm.T)[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
mu_shape = np.repeat(np.repeat(mus[np.newaxis,:], len(ww), axis=0)[np.newaxis, :, :], len(zz), axis=0)
phy_shape = np.repeat(np.repeat(phy[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
ww_shape = np.repeat(np.repeat(ww[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
zz_shape = np.repeat(np.repeat(zz[ :, np.newaxis], len(ww), axis=1)[:, :, np.newaxis], len(mus), axis=2)

# Initialaice the used tensors
II = plank_Ishape
QQ = II*0
II_new = II*1
QQ_new = QQ*0
#  ------------------- FUNCTIONS FOR THE SOLVE METHOD -------------------------------
# Function to compute the coeficients of the Short Characteristics method
def psi_calc(deltaum, deltaup, mode='quad'):

    U0 = 1 - np.exp(-deltaum)
    U1 = deltaum - U0
    U2 = deltaum**2 - 2*deltaum - 2 + 2*np.exp(-deltaum)
    
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

def RTE_SC_solve(II,QQ,SI,SQ,zz,mus, tau_z = 'imp'):
    for j in range(len(mus)):
        
        if tau_z == 'exp':
            taus = -(zz-np.min(zz))/mus[j]
        elif tau_z == 'imp':
            taus = np.exp(-zz)/mus[j]
        else:
            raise Exception('the way of computing tau(z,mu) should be exp or imp {} was introduced'.format(tau_z))
        
        deltau = abs(taus[1:] - taus[:-1])
        for i in range(1,len(zz)-1):

            psim,psio,psip = psi_calc(deltau[i-1], deltau[i])
            II_new[i,:,j] = II_new[i-1,:,j]*np.exp(-deltau[i-1]) + SI[i-1,:,j]*psim + SI[i,:,j]*psio + SI[i+1,:,j]*psip
            QQ_new[i,:,j] = QQ_new[i-1,:,j]*np.exp(-deltau[i-1]) + SQ[i-1,:,j]*psim + SQ[i,:,j]*psio + SQ[i+1,:,j]*psip

        psim, psio = psi_calc(deltau[-2], deltau[-1], mode='linear')
        II_new[-1,:,j] = II_new[-2,:,j]*np.exp(-deltau[-1]) + SI[-2,:,j]*psim + SI[-1,:,j]*psio 
        QQ_new[-1,:,j] = QQ_new[-2,:,j]*np.exp(-deltau[-1]) + SQ[-2,:,j]*psim + SQ[-1,:,j]*psio
    
    return II_new,QQ_new

# ------------- TEST ON THE SHORT CHARACTERISTICS METHOD ----------------------------
# We define the ilumination just at the bottom boundary
II = II*0
II[0] = plank_Ishape[0]
# Define the new vectors as the old ones
II_new = II*1
QQ_new = QQ*0
# Define the source function as a constant value with the dimensions of II
SI = 0.5*(plank_Ishape/plank_Ishape)
SQ = 0.25*(plank_Ishape/plank_Ishape)

II_new, QQ_new = RTE_SC_solve(II,QQ,SI,SQ,zz,mus, 'exp')

plt.plot(ww, II[0,:,-1])
plt.plot(ww, II_new[10, :,-1])
plt.show()
plt.plot(zz,II[:,0,-1])
plt.plot(zz,QQ_new[:,0,-1])
plt.plot(zz,II_new[:,0,-1])
plt.show()

Ip = II*np.exp(-((zz_shape-np.min(zz_shape))/mu_shape)) + 0.5*(1-np.exp(-(zz_shape-np.min(zz_shape)/mu_shape)))
Im = QQ*np.exp(-((zz_shape-np.min(zz_shape))/mu_shape)) + 0.25*(1-np.exp(-(zz_shape-np.min(zz_shape)/mu_shape)))
Ip_calc = II_new
Im_calc = QQ_new

plt.plot(zz, Ip[:, 50, -1], 'b', label='$I_{+}$')
plt.plot(zz, Im[:, 50, -1], 'b--', label='$I_{-}$')
plt.plot(zz, Ip_calc[:, 50, -1], 'r', label='$I_{+, calc}$')
plt.plot(zz, Im_calc[:, 50, -1], 'r--', label='$I_{-, calc}$')
plt.plot(zz, Ip[:, 50, -1], 'bx', label='$I_{+}$')
plt.plot(zz, Im[:, 50, -1], 'bx', label='$I_{-}$')
plt.plot(zz, Ip_calc[:, 50, -1], 'rx', label='$I_{+, calc}$')
plt.plot(zz, Im_calc[:, 50, -1], 'rx', label='$I_{-, calc}$')
plt.legend()
plt.show()

# -----------------------------------------------------------------------------------
# ---------------------- MAIN LOOP TO OBTAIN THE SOLUTION ---------------------------
# -----------------------------------------------------------------------------------

# Compute the source function as a tensor in of zz, ww, mus
S00 = 0*plank_Ishape
S00[0] = pm.eps*plank_Ishape[0]
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

w2jujl = (-1)**(1+pm.ju+pm.jl) * np.sqrt(3*(2*pm.ju + 1)) * jsymbols.j3(1, 1, 2, pm.ju, pm.ju, pm.jl)

# while max(diff) > pm.tolerance:
for i in range(10):
    # ----------------- SOLVE RTE BY THE SHORT CHARACTERISTICS ---------------------------
    print('Solving the Radiative Transpor Equations')
    II_new, QQ_new = RTE_SC_solve(II,QQ,SI,SQ,zz,mus, 'imp')

    # ---------------- COMPUTE THE COMPONENTS OF THE RADIATIVE TENSOR ----------------------
    print('computing the components of the radiative tensor')

    Jm00 = 1/2 * integ.simps(integ.simps(phy_shape*II_new))
    Jm02 = 1/np.sqrt(4**2 * 2) * integ.simps(
                                 integ.simps(
                                     phy_shape * (3*mu_shape**2 - 1)*II_new + 3*(mu_shape**2 - 1)*QQ_new ))

    # print('The Jm00 component is: {} and the Jm02 is: {}'.format(Jm00[-1],Jm02[-1]))
    # ---------------- COMPUTE THE SOURCE FUNCTIONS TO SOLVE THE RTE -----------------------
    print('Computing the source function to close the loop and solve the ETR again')

    Jm00_shape = np.repeat(np.repeat(Jm00[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)
    Jm02_shape = np.repeat(np.repeat(Jm02[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)

    S00 = (1-pm.eps)*Jm00_shape + pm.eps*plank_Ishape
    S20 = pm.Hd * (1-pm.eps)/(1 + (1-pm.eps)*pm.dep_col) * w2jujl**2 * Jm02_shape

    SLI = S00 + w2jujl * (3*mu_shape**2 - 1)/np.sqrt(8) * S20
    SLQ = w2jujl * 3*(mu_shape**2 - 1)/np.sqrt(8) * S20

    SI = phy_shape/(phy_shape + pm.r)*SLI + pm.r/(phy_shape + pm.r)*plank_Ishape
    SQ = phy_shape/(phy_shape + pm.r)*SLQ

    print('Computing the differences and reasign the intensities')
    diff = np.append(II - II_new, QQ - QQ_new)
    if( np.all( diff < pm.tolerance ) ):
        print('------ FINISHED!!-----')
        break

    II = II_new
    QQ = QQ_new

plt.plot(ww, II[0,:,int(50)], color='k', label='Intensity')
plt.plot(ww, SI[0,:,50], color='b', label='Source function')
# plt.plot(ww, SLI[0,:,50], color='r', label='line source function')
plt.legend()
plt.show()
plt.plot(zz,II[:,0,0], color='k', label='$I(z)$')
plt.plot(zz,II_new[:,0,0], color='b', label='$I_{new}(z)$')
plt.plot(zz,SI[:,0,0], color = 'r', label = '$S_I$')
plt.legend()
plt.show()

# tolerancia en todas las capas en SO0, S02

# -------------------- ONCE WE OBTAIN THE SOLUTION, COMPUTE THE POPULATIONS ----------------
Jm00_shape = np.repeat(np.repeat(Jm00[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)
Jm02_shape = np.repeat(np.repeat(Jm02[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)

rho00 = np.sqrt((2*pm.ju + 1)/(2*pm.jl+1)) * (2*cte.h*ww_shape**3/cte.c**2)**-1 * \
    ((1-pm.eps)*Jm00_shape + pm.eps*plank_Ishape)/((1-pm.eps)*pm.dep_col + 1)

rho02 = np.sqrt((2*pm.ju + 1)/(2*pm.jl+1)) * (2*cte.h*ww_shape**3/cte.c**2)**-1 * \
    ((1-pm.eps)*Jm02_shape)/((1-pm.eps)*(1j*pm.Hd*2 + pm.dep_col) + 1)
