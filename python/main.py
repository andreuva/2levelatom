#############################################################################
#                2 LEVEL ATOM ATMOSPHERE SOLVER                             #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ
from tqdm import tqdm,trange
# local imports of constants parameters and functions
import constants as cte
import parameters as pm
import physical_functions as func
from jsymbols import jsymbols
import gaussian_quadrature as gauss
jsymbols = jsymbols()

# We define the z0, zl, dz as our heigt grid (just 1D because of a
# plane-parallel atmosfere and axial-simetry)
zz = np.arange(pm.zl, pm.zu + pm.dz, pm.dz)          # compute the 1D grid

# Define the grid in frequencies ( or wavelengths )
if pm.w_normaliced:
    ww = np.arange(pm.wl, pm.wu + pm.dw, pm.dw)          # compute the 1D grid
    wnorm = ww.copy()
else:
    ww = np.arange(pm.wl, pm.wu + pm.dw, pm.dw)          # Compute the 1D spectral grid
    wnorm = (ww - pm.w0)/pm.wa          # normalice the frequency to compute phy

# Define the directions of the rays
if pm.gaussian:
    [weigths, mus, err] = gauss.GaussLegendreWeights(pm.qnd, pm.tolerance)
    weigths_shape = np.repeat(np.repeat(weigths[np.newaxis,:], len(wnorm), axis=0)[np.newaxis, :, :], len(zz), axis=0)
else:
    if pm.qnd%2 != 0:
        print('Changing the number of directions in the cuadrature', end=' ')
        print(f'from {pm.qnd}',end=' ')
        pm.qnd += 1
        print(f'to {pm.qnd}')

    if pm.qnd < 50:
        print('To few ray directions, changing', end=' ')
        print(f'from {pm.qnd}',end=' ')
        pm.qnd = 60
        print(f'to {pm.qnd}')
        
    mus = np.linspace(-1, 1, pm.qnd)

# ------------------------ SOME INITIAL CONDITIONS --------------------------
# Compute the initial Voigts vectors
phy = np.zeros_like(wnorm)
for i in range(len(wnorm)):
    phy[i] = np.real(func.voigt(wnorm[i], pm.a))
phy = phy/integ.simps(phy, wnorm)          # normalice phy to sum 1

# Initialaice the intensities vectors to solve the ETR
# Computed as a tensor in zz, ww, mus
plank_Ishape = np.repeat(np.repeat(np.ones_like(wnorm)[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
mu_shape = np.repeat(np.repeat(mus[np.newaxis,:], len(wnorm), axis=0)[np.newaxis, :, :], len(zz), axis=0)
phy_shape = np.repeat(np.repeat(phy[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
wnorm_shape = np.repeat(np.repeat(wnorm[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
zz_shape = np.repeat(np.repeat(zz[ :, np.newaxis], len(wnorm), axis=1)[:, :, np.newaxis], len(mus), axis=2)
tau_shape = np.exp(-zz_shape)*phy_shape


w2jujl = 1 # jsymbols.j6(1,1,2,1,1,0)/jsymbols.j6(1,1,0,1,1,0)
rr = phy_shape/(phy_shape + pm.r)

# Compute the source function as a tensor in of zz, ww, mus
# Initialaice the used tensors
II = plank_Ishape.copy()
II[1:] = II[1:]*0
QQ = np.zeros_like(II)

S00 = plank_Ishape.copy()
S20 = np.zeros_like(S00)

SLI = S00 + w2jujl * (3*mu_shape**2 - 1)/np.sqrt(8) * S20
SLQ = w2jujl * 3*(mu_shape**2 - 1)/np.sqrt(8) * S20

SI = rr*SLI + (1 - rr)*plank_Ishape
SQ = rr*SLQ

SI_analitic = (1-pm.eps)*(1-np.exp(-tau_shape*np.sqrt(3*pm.eps))/(1+np.sqrt(pm.eps))) + pm.eps*plank_Ishape

if pm.initial_plots:
    plt.plot(wnorm, (II/plank_Ishape)[5,:,-1], color='k', label=r'$B_{\nu}(T= $'+'{}'.format(pm.T) + '$)$')
    plt.plot(wnorm, (QQ/II)[5,:,-1], color='g', label=r'$Q(\nu,z=0,\mu=1)$')
    plt.plot(wnorm, (SI/plank_Ishape)[5,:,-1], color='b', label=r'$S_I(\nu,z=0,\mu=1)$')
    plt.plot(wnorm, (SLI//plank_Ishape)[5,:,-1], color='r', label=r'$S^L_I(\nu,z=0,\mu=1)$')
    plt.xlabel(r'$\nu\ (Hz)$')
    plt.legend()
    plt.show()
    plt.plot(wnorm, (phy/(phy + pm.r)), color='r', label= r'$ \dfrac{\phi(\nu)}{\phi(\nu) + r}$')
    plt.plot(wnorm, pm.r/(phy + pm.r), color='b', label=r'$ \dfrac{r}{\phi(\nu) + r}$')
    plt.plot(wnorm, phy_shape[0,:,0], color='k', label=r'$ \phi(\nu) $')
    plt.xlabel(r'$\nu\ (Hz)$'); plt.title('profiles with $a=${} and $w_0=${:.3e} Hz'.format(pm.a,pm.w0))
    plt.legend()
    plt.show()

#  ------------------- FUNCTIONS FOR THE SOLVE METHOD --------------------------


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



def RTE_SC_solve(I, Q, SI, SQ, tau, mu):
    """
    Compute the new intensities form the source function with the SC method
    """

    l_st = np.zeros_like(I)

    for j in range(len(mu)):
        if mu[j] > 0:
            psip_prev = 0
            for i in range(1, tau.shape[0]):
                deltaum = np.abs((tau[i-1,:]-tau[i,:]) / mu[j])

                if (i < (len(tau)-1)):
                    deltaup = np.abs((tau[i,:]-tau[i+1,:]) / mu[j])
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
            for i in range(tau.shape[0]-2,-1,-1):
                deltaum = np.abs((tau[i,:]-tau[i+1,:])/mu[j])

                if (i > 0):
                    deltaup = np.abs((tau[i-1,:]-tau[i,:])/mu[j])

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
    
    return I, Q, l_st

# -----------------------------------------------------------------------------------
# ---------------------- MAIN LOOP TO OBTAIN THE SOLUTION ---------------------------
# -----------------------------------------------------------------------------------
t = trange(pm.max_iter, desc='Iterations:', leave=True)
for itt in t:
    # ----------------- SOLVE RTE BY THE SHORT CHARACTERISTICS ---------------------------
    # print('Solving the Radiative Transpor Equations')
    II, QQ, lambd = RTE_SC_solve(II, QQ, SI, SQ, tau_shape[:,:,-1], mus)
    
    if np.min(II) < -1e-6:
        print('found a negative intensity, stopping')
        break

    # ---------------- COMPUTE THE COMPONENTS OF THE RADIATIVE TENSOR ----------------------
    # print('computing the components of the radiative tensor')

    Jm00 = 1/2. * np.sum(II*weigths_shape, axis=-1)
    Jm00 = integ.simps(phy_shape[:,:,0]*Jm00, wnorm)

    Jm02 = (3*mu_shape**2 - 1)*II + 3*(mu_shape**2 - 1)*QQ
    Jm02 = 1/np.sqrt(4**2 * 2) * np.sum(Jm02*weigths_shape, axis=-1)
    Jm02 =  integ.simps( phy_shape[:,:,0]*Jm02, wnorm)

    lambd = 1/2. * np.sum(lambd*weigths_shape, axis=-1)
    lambd = integ.simps( phy_shape[:,:,0] * lambd, wnorm)

    Jm00_shape = np.repeat(np.repeat(Jm00[ :, np.newaxis], len(wnorm), axis=1)[ :, :, np.newaxis], len(mus), axis=2)
    Jm02_shape = np.repeat(np.repeat(Jm02[ :, np.newaxis], len(wnorm), axis=1)[ :, :, np.newaxis], len(mus), axis=2)
    lambd = np.repeat(np.repeat(lambd[ :, np.newaxis], len(wnorm), axis=1)[ :, :, np.newaxis], len(mus), axis=2)

    # ---------------- COMPUTE THE SOURCE FUNCTIONS TO SOLVE THE RTE -----------------------
    # print('Computing the source function to close the loop and solve the ETR again')
    # computing Jm00 and Jm02 with tensor shape as the rest of the variables

    S00 = (1-pm.eps)*Jm00_shape + pm.eps*plank_Ishape
    S20 = pm.Hd * (1-pm.eps)/(1 + (1-pm.eps)*pm.dep_col) * w2jujl * Jm02_shape

    SLI = S00 + w2jujl * (3*mu_shape**2 - 1)/np.sqrt(8) * S20
    SLQ = w2jujl * 3*(mu_shape**2 - 1)/np.sqrt(8) * S20

    SI_new = rr*SLI + (1 - rr)*plank_Ishape
    SQ_new = rr*SLQ

    # Applying the lambda operator to accelerate the convergence
    # SI_new = (SI_new - SI)/(1 - (1-pm.eps)*lambd) + SI

    # print('Computing the differences and reasign the intensities')
    olds = np.array([SI])
    news = np.array([SI_new])
    SI = SI_new.copy()
    SQ = SQ_new.copy()
    tol = np.max(np.abs(np.abs(olds - news)/(olds+1e-200)))
    # print('Actual tolerance is :',tol)
    t.set_description('Actual tolerance is : %1.3e' % tol)
    t.refresh() # to show immediately the update

    if( tol < pm.tolerance ):
        print('-------------- FINISHED!!---------------')
        break

if (itt >= pm.max_iter - 1):
    print('Ops! The solution with the desired tolerance has not been found')
    print('Although an aproximate solution may have been found. Try to change')
    print('the parameters to obtain an optimal solution.')
    print('The found tolerance is: ',tol*100, '%')



plt.plot(wnorm, (II/plank_Ishape)[-1, :, -1], 'b', label='$I$')
plt.plot(wnorm, (QQ/II)[-1, :, -1], 'r', label='$Q/I$')
plt.plot(wnorm, (SI/plank_Ishape)[-1,:,-1], 'm', label=r'$S_I/B_{\nu}$')
plt.plot(wnorm, (SQ/SI)[-1,:,-1], 'k', label=r'$S_Q/S_I$')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()
plt.plot(zz, (II/plank_Ishape)[:, pm.nn, -1], 'r', label=r'$I(\mu=1)$')
plt.plot(zz, (SI/plank_Ishape)[:, pm.nn, -1], 'k', label=r'$S_I(\mu=1)$')
# plt.plot(zz, (QQ/II)[:, pm.nn, -1], 'b', label=r'$Q(\mu=1)$')
plt.plot(zz,(Jm00_shape/plank_Ishape)[:,pm.nn,-1], 'g', label=r'$J^0_0/B_\nu(\mu=1)$')
# plt.plot(zz,(Jm02_shape/plank_Ishape)[:,pm.nn,-1], 'm', label=r'$J^2_0/B_\nu(\mu=1)$')
plt.plot(zz, (II/plank_Ishape)[:, pm.nn, 0], 'r--', label=r'$I/B_{\nu}(\mu=-1)$')
plt.plot(zz, (SI/plank_Ishape)[:, pm.nn, 0], 'k--', label=r'$S_I(\mu=-1)$')
# plt.plot(zz, (QQ/II)[:, pm.nn, 0], 'b--', label=r'$Q/I(\mu=-1)$')
plt.plot(zz,(Jm00_shape/plank_Ishape)[:,pm.nn,0], 'g--', label=r'$J^0_0/B_\nu(\mu=-1)$')
# plt.plot(zz,(Jm02_shape/plank_Ishape)[:,pm.nn,0], 'm--', label=r'$J^2_0/B_\nu(\mu=-1)$')
plt.plot(zz,(SI_analitic)[:,pm.nn,-1], 'pink', label = 'Analitic solution')
plt.legend(); plt.xlabel('z')
plt.show()

plt.imshow(II[:, :, pm.mm], origin='lower', aspect='auto'); plt.xlabel(r'$\nu$'); plt.ylabel('z'); plt.title('$I_{calc}$');plt.colorbar(); plt.show()
plt.imshow(QQ[:, :, pm.mm], origin='lower', aspect='auto'); plt.xlabel(r'$\nu$'); plt.ylabel('z'); plt.title('$Q_{calc}$');plt.colorbar(); plt.show()
plt.imshow(SI[:,:,pm.mm], origin='lower', aspect='auto'); plt.title(r'$S_I$');plt.colorbar(); plt.show()
plt.imshow(SQ[:,:,pm.mm], origin='lower', aspect='auto'); plt.title(r'$S_Q$');plt.colorbar(); plt.show()
plt.imshow((Jm00_shape/plank_Ishape)[:,:,pm.mm], origin='lower', aspect='auto'); plt.xlabel(r'$\nu$'); plt.ylabel('z'); plt.title(r'$J^0_0/B_\nu$');plt.colorbar(); plt.show()
plt.imshow((Jm02_shape/plank_Ishape)[:,:,pm.mm], origin='lower', aspect='auto'); plt.xlabel(r'$\nu$'); plt.ylabel('z'); plt.title(r'$J^2_0/B_\nu$');plt.colorbar(); plt.show()
plt.imshow(II[:, pm.nn, :], origin='lower', aspect='auto'); plt.title('$I$'); plt.colorbar(); plt.show()
plt.imshow(QQ[:, pm.nn, :], origin='lower', aspect='auto'); plt.title('$Q$'); plt.colorbar(); plt.show()
plt.imshow(SI[:,pm.nn,:], origin='lower', aspect='auto'); plt.title('$S_I$');plt.colorbar(); plt.show()
plt.imshow(SQ[:,pm.nn,:], origin='lower', aspect='auto'); plt.title('$S_Q$');plt.colorbar(); plt.show()
plt.imshow((Jm00_shape/plank_Ishape)[:,pm.nn,:], origin='lower', aspect='auto'); plt.title(r'$J^0_0/B_\nu$');plt.colorbar(); plt.show()
plt.imshow((Jm02_shape/plank_Ishape)[:,pm.nn,:], origin='lower', aspect='auto'); plt.title(r'$J^2_0/B_\nu$');plt.colorbar(); plt.show()