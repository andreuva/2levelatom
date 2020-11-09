#############################################################################
#                2 LEVEL ATOM ATMOSPHERE SOLVER                             #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ
from tqdm import tqdm
# local imports of constants parameters and functions
import constants as cte
import parameters as pm
import physical_functions as func

# We define the z0, zl, dz as our heigt grid (just 1D because of a
# plane-parallel atmosfere and axial-simetry)
zz = np.arange(pm.zl, pm.zu + pm.dz, pm.dz)          # compute the 1D grid

# Define the grid in frequencies ( or wavelengths )
ww = np.arange(pm.wl, pm.wu, pm.dw)          # Compute the 1D spectral grid

# Define the directions of the rays
if pm.qnd%2 != 0:
    print('Changing the number of directions in the cuadrature', end=' ')
    print(f'from {pm.qnd}',end=' ')
    pm.qnd += 1
    print(f'to {pm.qnd}')
    
''' mus = np.linspace(-1, 1, pm.qnd) '''
mus = np.array([-1/np.sqrt(3) , 1/np.sqrt(3)])
tau = np.exp(-zz)

# Initialaice the basic grid and auxiliar tensors
plank_Ishape = np.repeat(np.repeat(func.plank_wien(ww, pm.T)[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
mu_shape = np.repeat(np.repeat(mus[np.newaxis,:], len(ww), axis=0)[np.newaxis, :, :], len(zz), axis=0)
ww_shape = np.repeat(np.repeat(ww[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
zz_shape = np.repeat(np.repeat(zz[ :, np.newaxis], len(ww), axis=1)[:, :, np.newaxis], len(mus), axis=2)
tau_shape = np.exp(-zz_shape)

# Compute the source function, intensities and polarizations as a tensor in zz, ww, mus
# Initialaice the used tensors
II = plank_Ishape.copy()
II[1:] = II[1:]*0
QQ = np.zeros_like(II)

SI = plank_Ishape.copy()
SQ = np.zeros_like(SI)                                           # SQ = 0 (size of SI)

SI_analitic = (1-pm.eps)*(1-np.exp(-tau_shape*np.sqrt(3*pm.eps))/(1+np.sqrt(pm.eps))) + pm.eps*plank_Ishape
error = []
mrc = []

#  ------------------- FUNCTIONS FOR THE SOLVE METHOD --------------------------
# Function to compute the coeficients of the Short Characteristics method
def psi_calc(deltaum, deltaup, mode='quadratic'):
    """
    Compute of the psi coefficients in the SC method
    """
    U0 = 1 - np.exp(-deltaum)
    U1 = deltaum - U0
    
    if mode == 'quadratic':
        U2 = (deltaum)**2 - 2*U1

        psim = U0 + (U2 - U1*(deltaup + 2*deltaum))/(deltaum*(deltaum + deltaup))
        psio = (U1*(deltaum + deltaup) - U2)/(deltaum*deltaup)
        psip = (U2 - U1*deltaum)/(deltaup*(deltaup+deltaum))
        return psim, psio, psip
    elif mode == 'lineal':
        psim = U0 - U1/deltaum
        psio = U1/deltaum
        return psim, psio
    else:
        raise Exception(f'mode should be quadratic or lineal but {mode} was introduced')

def RTE_SC_solve(I,Q,SI,SQ,tau,mu):
    """
    Compute the new intensities form the source function with the SC method
    """

    l_st = np.zeros_like(I)

    for j in range(len(mu)):
        if mu[j] > 0:
            psip_prev = 0
            for i in range(1,len(tau)):
                deltaum = (tau[i-1]-tau[i])/mu[j]

                if (i < (len(tau)-1)):
                    deltaup = np.abs((tau[i]-tau[i+1])/mu[j])
                    psim,psio,psip = psi_calc(deltaum, deltaup)
                    I[i,:,j] = I[i-1,:,j]*np.exp(-deltaum) + SI[i-1,:,j]*psim + SI[i,:,j]*psio + SI[i+1,:,j]*psip
                    Q[i,:,j] = Q[i-1,:,j]*np.exp(-deltaum) + SQ[i-1,:,j]*psim + SQ[i,:,j]*psio + SQ[i+1,:,j]*psip
                    l_st[i,:,j] = psip_prev*np.exp(-deltaum) + psio  
                    psip_prev = psip
                else:
                    psim, psio = psi_calc(deltaum, deltaum, mode='lineal')
                    I[i,:,j] = I[i-1,:,j]*np.exp(-deltaum) + SI[i-1,:,j]*psim + SI[i,:,j]*psio
                    Q[i,:,j] = Q[i-1,:,j]*np.exp(-deltaum) + SQ[i-1,:,j]*psim + SQ[i,:,j]*psio
                    l_st[i,:,j] = psip_prev*np.exp(-deltaum) + psio
        else:
            psip_prev = 0
            for i in range(len(tau)-2,-1,-1):
                deltaum = -(tau[i]-tau[i+1])/mu[j]

                if (i > 0):
                    deltaup = np.abs((tau[i-1]-tau[i])/mu[j])

                    psim,psio,psip = psi_calc(deltaum, deltaup)
                    I[i,:,j] = I[i+1,:,j]*np.exp(-deltaum) + SI[i+1,:,j]*psim + SI[i,:,j]*psio + SI[i-1,:,j]*psip
                    Q[i,:,j] = Q[i+1,:,j]*np.exp(-deltaum) + SQ[i+1,:,j]*psim + SQ[i,:,j]*psio + SQ[i-1,:,j]*psip

                    l_st[i,:,j] = psip_prev*np.exp(-deltaum) + psio
                    psip_prev = psip
                else:
                    psim, psio = psi_calc(deltaum, deltaum, mode='lineal')
    
                    I[i,:,j] = I[i+1,:,j]*np.exp(-deltaum) + SI[i+1,:,j]*psim + SI[i,:,j]*psio 
                    Q[i,:,j] = Q[i+1,:,j]*np.exp(-deltaum) + SQ[i+1,:,j]*psim + SQ[i,:,j]*psio
                    l_st[i,:,j] = psip_prev*np.exp(-deltaum) + psio
    
    return I,Q, l_st

# -----------------------------------------------------------------------------------
# ---------------------- MAIN LOOP TO OBTAIN THE SOLUTION ---------------------------
# -----------------------------------------------------------------------------------
for itt in tqdm(range(1,pm.max_iter+1)):

    # ----------------- SOLVE RTE BY THE SHORT CHARACTERISTICS ---------------------------
    # print('Solving the Radiative Transpor Equations')
    II, QQ, lamb_st = RTE_SC_solve(II,QQ,SI,SQ,tau,mus)
    
    if np.min(II) < 0:
        print('I<0 aborting')
        break

    # ---------------- COMPUTE THE COMPONENTS OF THE RADIATIVE TENSOR ----------------------
    # print('computing the components of the radiative tensor')
    Jm00 = 1/2 * (II[:,:,0] + II[:,:,1])
    Jm02 = ( 3*mu_shape**2 - 1)*II + 3*(mu_shape**2 - 1)*QQ
    Jm02 = 1/np.sqrt(4**2 * 2)  * (Jm02[:,:,0] + Jm02[:,:,1])
    lamb_st = 1/2 * (lamb_st[:,:,0] + lamb_st[:,:,1])

    # computing lambda, Jm00 and Jm02 with tensor shape as the rest of the variables
    Jm00_shape = np.repeat(Jm00[ :, :, np.newaxis], len(mus), axis=2)
    Jm02_shape = np.repeat(Jm02[ :, :, np.newaxis], len(mus), axis=2)
    lamb_st = np.repeat(lamb_st[ :, :, np.newaxis], len(mus), axis=2)

    # ---------------- COMPUTE THE SOURCE FUNCTIONS TO SOLVE THE RTE -----------------------
    # print('Computing the source function to close the loop and solve the ETR again')

    SI_new = (1-pm.eps)*Jm00_shape + pm.eps*plank_Ishape
    SQ_new = (1-pm.eps)*Jm02_shape

    # Applying the lambda operator to accelerate the convergence
    SI_new = (SI_new - SI)/(1 - (1-pm.eps)*lamb_st) + SI

    # print('Computing the differences and reasign the intensities')
    tol = np.max(np.abs(np.abs(SI - SI_new)/(SI+1e-200)))
    err = np.max(np.abs(SI_analitic - SI/plank_Ishape)[:,:,1])
    mrc.append(tol)
    error.append(err)
    
    if( tol < pm.tolerance ):
        print('-------------------- FINISHED!!---------------------')
        print(f' After {itt} iterations with a tolerance of {tol}')
        break
    
    SI = np.copy(SI_new)
    SQ = np.copy(SQ_new)

if (itt >= pm.max_iter - 1):
    print('Ops! The solution with the desired tolerance has not been found')
    print('Although an aproximate solution may have been found. Try to change')
    print('the parameters to obtain an optimal solution.')
    print('The found tolerance is: ',tol*100)

# Plotting if necesary
if pm.plots:
    plt.plot(zz, (II/plank_Ishape)[:, pm.nn, -1], 'b', label='$I$')
    plt.plot(zz, (QQ/plank_Ishape)[:, pm.nn, -1], 'r--', label='$Q$')
    plt.legend(); plt.xlabel('z')
    plt.show()
    plt.plot(zz,(Jm00_shape/plank_Ishape)[:,pm.nn,-1], 'b--', label=r'$J^0_0/B_\nu$')
    plt.plot(zz,(Jm02_shape/plank_Ishape)[:,pm.nn,-1], 'r-.', label=r'$J^2_0/B_\nu$')
    plt.legend(); plt.show()
    plt.plot(mus,(Jm00_shape/plank_Ishape)[1,pm.nn,:], 'b--', label=r'$J^0_0/B_\nu$')
    plt.plot(mus,(Jm02_shape/plank_Ishape)[1,pm.nn,:], 'r-.', label=r'$J^2_0/B_\nu$')
    plt.legend(); plt.show()

plt.imshow((II)[:, pm.nn, :], origin='lower', aspect='auto'); plt.xlabel('$\mu$'); plt.ylabel('z'); plt.title('$I$'); plt.colorbar(); plt.show()
plt.imshow((QQ)[:, pm.nn, :], origin='lower', aspect='auto'); plt.xlabel('$\mu$'); plt.ylabel('z'); plt.title('$Q$'); plt.colorbar(); plt.show()
plt.imshow((SI)[:,pm.nn,:], origin='lower', aspect='auto'); plt.xlabel('$\mu$'); plt.ylabel('z'); plt.title(r'$S_I$');plt.colorbar(); plt.show()
plt.imshow((SQ)[:,pm.nn,:], origin='lower', aspect='auto'); plt.xlabel('$\mu$'); plt.ylabel('z'); plt.title(r'$S_Q$');plt.colorbar(); plt.show()
plt.imshow((Jm00_shape/plank_Ishape)[:,pm.nn,:], origin='lower', aspect='auto'); plt.xlabel('$\mu$'); plt.ylabel('z'); plt.title(r'$J^0_0/B_\nu$');plt.colorbar(); plt.show()
plt.imshow((Jm02_shape/plank_Ishape)[:,pm.nn,:], origin='lower', aspect='auto'); plt.xlabel('$\mu$'); plt.ylabel('z'); plt.title(r'$J^2_0/B_\nu$');plt.colorbar(); plt.show()

plt.plot(zz,(II/plank_Ishape)[:,pm.nn,0], 'k--', label=r'$I/B_{\nu}$ upward')
plt.plot(zz,(II/plank_Ishape)[:,pm.nn, -1], 'k:', label=r'$I/B_{\nu}$ downward')
plt.plot(zz,(QQ)[:,pm.nn,-1], 'g', label=r'$Q/I$')
plt.plot(zz,(SI_new/plank_Ishape)[:,pm.nn,-1], 'r--', label = r'$S_I/B_{\nu}$')
plt.plot(zz,(SQ/SI)[:,pm.nn,-1], 'g--', label = r'$S_Q/S_I$')
# plt.plot(zz,(Jm00_shape/plank_Ishape)[:,pm.nn,-1], 'r', label=r'$J^0_0/B_\nu$ shape')
# plt.plot(zz,(Jm02_shape/plank_Ishape)[:,pm.nn,-1], 'r--', label=r'$J^2_0/B_\nu$ shape')
plt.plot(zz,(SI_analitic)[:,pm.nn,-1], 'pink', label = 'Analitic solution')
plt.legend()
plt.show()

plt.plot(np.log10(error),'--', label=f'Si - Si(sol) dz={pm.dz}')
plt.plot(np.log10(mrc), label=f'MRC dz={pm.dz}')
plt.xscale('log')
plt.show()