#############################################################################
#                2 LEVEL ATOM ATMOSPHERE SOLVER                             #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ
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

# ------------------------ SOME INITIAL CONDITIONS --------------------------
# Compute the initial Voigts vectors
phy = np.zeros_like(ww)
wnorm = (ww - pm.w0)/pm.wa          # normalice the frequency to compute phy
for i in range(len(wnorm)):
    phy[i] = np.real(func.voigt(wnorm[i], pm.a))
phy = phy/integ.simps(phy, wnorm)          # normalice phy to sum 1

# Initialaice the intensities vectors to solve the ETR
# Computed as a tensor in zz, ww, mus
plank_Ishape = np.repeat(np.repeat(func.plank_wien(ww, pm.T)[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
mu_shape = np.repeat(np.repeat(mus[np.newaxis,:], len(ww), axis=0)[np.newaxis, :, :], len(zz), axis=0)
phy_shape = np.repeat(np.repeat(phy[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
ww_shape = np.repeat(np.repeat(ww[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
zz_shape = np.repeat(np.repeat(zz[ :, np.newaxis], len(ww), axis=1)[:, :, np.newaxis], len(mus), axis=2)

#  ------------------- FUNCTIONS FOR THE SOLVE METHOD --------------------------
# Function to compute the coeficients of the Short Characteristics method
def psi_calc(deltaum, deltaup, mode='quad'):

    U0 = 1 - np.exp(-deltaum)
    U1 = deltaum - U0
    U2 = (deltaum)**2 - 2*U1
    
    if mode == 'quad':
        psim = U0 + (U2 - U1*(deltaup + 2*deltaum))/(deltaum*(deltaum + deltaup))
        psio = (U1*(deltaum + deltaup) - U2)/(deltaum*deltaup)
        psip = (U2 - U1*deltaum)/(deltaup*(deltaup+deltaum))
        return psim, psio, psip
    elif mode == 'linear':
        psim = U0 - U1/deltaum
        psio = U1/deltaum
        return psim, psio
    else:
        raise Exception('mode should be quad or lineal but {} was introduced'.format(mode))

def RTE_SC_solve(I,Q,SI,SQ,zz,mus, tau_z = 'imp'):
    I_new = np.copy(I)
    Q_new = np.copy(Q)
    for j in range(len(mus)):
        
        if tau_z == 'exp':
            taus = -(zz-np.min(zz))/mus[j]
        elif tau_z == 'imp':
            taus = np.exp(-zz)/mus[j]
        else:
            raise Exception('the way of computing tau(z,mu) should be exp or imp {} was introduced'.format(tau_z))
        
        deltau = np.abs(taus[1:] - taus[:-1])
        for i in range(1,len(zz)-1):

            psim,psio,psip = psi_calc(deltau[i-1], deltau[i])
            I_new[i,:,j] = I_new[i-1,:,j]*np.exp(-deltau[i-1]) + SI[i-1,:,j]*psim + SI[i,:,j]*psio + SI[i+1,:,j]*psip
            Q_new[i,:,j] = Q_new[i-1,:,j]*np.exp(-deltau[i-1]) + SQ[i-1,:,j]*psim + SQ[i,:,j]*psio + SQ[i+1,:,j]*psip
            if np.min(I_new[i,:,j]) < 0:
                print(np.unravel_index(np.argmin(I_new), I_new.shape))
                print(i,j, np.min(I_new[i,:,j]))
                print(psim,psio,psip)
                print(deltau[i-1], deltau[i])

                plt.plot(ww, I_new[i-1,:,j])
                plt.plot(ww, I_new[i,:,j])
                plt.show()
                exit()

        psim, psio = psi_calc(deltau[-2], deltau[-1], mode='linear')
        I_new[-1,:,j] = I_new[-2,:,j]*np.exp(-deltau[-1]) + SI[-2,:,j]*psim + SI[-1,:,j]*psio 
        Q_new[-1,:,j] = Q_new[-2,:,j]*np.exp(-deltau[-1]) + SQ[-2,:,j]*psim + SQ[-1,:,j]*psio
    
    return I_new,Q_new


# ------------- TEST ON THE SHORT CHARACTERISTICS METHOD ------------------------
# We define the ilumination just at the bottom boundary
# Initialaice the used tensors

II = np.zeros_like(plank_Ishape)
QQ = np.zeros_like(II)
II[0] = plank_Ishape[0]
# Define the new vectors as the old ones
II_new = np.copy(II)
QQ_new = np.copy(QQ)
# Define the source function as a constant value with the dimensions of II
SI = 0.5*np.ones_like(II)
SQ = 0.25*np.ones_like(QQ)

II_new, QQ_new = RTE_SC_solve(II_new,QQ_new,SI,SQ,zz,mus, 'exp')

II = II*np.exp(-((zz_shape-np.min(zz_shape))/mu_shape)) + 0.5*(1-np.exp(-(zz_shape-np.min(zz_shape)/mu_shape)))
QQ = QQ*np.exp(-((zz_shape-np.min(zz_shape))/mu_shape)) + 0.25*(1-np.exp(-(zz_shape-np.min(zz_shape)/mu_shape)))

plt.plot(zz, II[:, 50, -1], 'b', label='$I$')
plt.plot(zz, QQ[:, 50, -1], 'b--', label='$Q$')
plt.plot(zz, II_new[:, 50, -1], 'rx', label='$I_{calc}$')
plt.plot(zz, QQ_new[:, 50, -1], 'rx', label='$Q_{calc}$')
plt.legend()
plt.show()

# -----------------------------------------------------------------------------------
# ---------------------- MAIN LOOP TO OBTAIN THE SOLUTION ---------------------------
# -----------------------------------------------------------------------------------
# Compute the source function as a tensor in of zz, ww, mus
# Initialaice the used tensors
II = np.copy(plank_Ishape)
# II[0] = plank_Ishape[0]
QQ = np.zeros_like(II)
II_new = np.copy(II)
QQ_new = np.zeros_like(QQ)


S00 = pm.eps*plank_Ishape
SLI = np.copy(S00)
SLQ = np.zeros_like(SLI)
SI = phy_shape/(phy_shape + pm.r) * SLI + (pm.r/(phy_shape + pm.r)) * plank_Ishape
SQ = np.zeros_like(SI)                                           # SQ = 0 (size of SI)

plt.plot(ww, II[50,:,-1], color='k', label=r'$B_{\nu}(T= $'+'{}'.format(pm.T) + '$)$')
plt.plot(ww, QQ[50,:,-1], color='g', label=r'$Q(\nu,z=0,\mu=1)$')
plt.plot(ww, SI[50,:,-1], color='b', label=r'$S_I(\nu,z=0,\mu=1)$')
plt.plot(ww, SLI[50,:,-1], color='r', label=r'$S^L_I(\nu,z=0,\mu=1)$')
plt.xlabel(r'$\nu\ (Hz)$')
plt.legend()
plt.show()
plt.plot(ww, (phy/(phy + pm.r)), color='r', label= r'$ \dfrac{\phi(\nu)}{\phi(\nu) + r}$')
plt.plot(ww, pm.r/(phy + pm.r), color='b', label=r'$ \dfrac{r}{\phi(\nu) + r}$')
plt.plot(ww, phy_shape[0,:,0], color='k', label=r'$ \phi(\nu) $')
plt.xlabel(r'$\nu\ (Hz)$'); plt.title('profiles with $a=${} and $w_0=${:.3e} Hz'.format(pm.a,pm.w0))
plt.legend()
plt.show()

w2jujl = jsymbols.j6(1,1,2,1,1,0)/jsymbols.j6(1,1,0,1,1,0)

for i in range(pm.max_iter):

    # ----------------- SOLVE RTE BY THE SHORT CHARACTERISTICS ---------------------------
    print('Solving the Radiative Transpor Equations')
    II_new, QQ_new = RTE_SC_solve(II,QQ,SI,SQ,zz,mus, 'imp')
    
    # plt.plot(ww, II[-1, :, -1], 'b', label='$I$')
    # plt.plot(ww, II_new[-1, :, -1], 'b-.', label='$I_{calc}$')
    # plt.plot(ww, QQ[-1, :, -1], 'r--', label='$Q$')
    # plt.plot(ww, QQ_new[-1, :, -1], 'r.', label='$Q_{calc}$')
    # plt.plot(ww, SI[-1,:,-1], color='k', label=r'$S_I(\nu,z= ,\mu=1)$')
    # plt.plot(ww, SLI[-1,:,-1], color='g', label=r'$S^L_I(\nu,z= ,\mu=1)$')
    # plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
    # plt.show()
    # plt.plot(zz, II[:, 50, -1], 'b', label='$I$')
    # plt.plot(zz, QQ[:, 50, -1], 'r--', label='$Q$')
    # plt.plot(zz, II_new[:, 50, -1], 'b-.', label='$I_{calc}$')
    # plt.plot(zz, QQ_new[:, 50, -1], 'r.', label='$Q_{calc}$')
    # plt.legend(); plt.xlabel('z')
    # plt.show()

    # plt.imshow(II[:, :, -1], origin='lower', aspect='equal'); plt.title('$I$'); plt.colorbar(); plt.show()
    # plt.imshow(II_new[:, :, -1], origin='lower', aspect='equal'); plt.title('$I_{calc}$');plt.colorbar(); plt.show()
    # plt.imshow(QQ[:, :, -1], origin='lower', aspect='equal'); plt.title('$Q$'); plt.colorbar(); plt.show()
    # plt.imshow(QQ_new[:, :, -1], origin='lower', aspect='equal'); plt.title('$Q_{calc}$');plt.colorbar(); plt.show()

    # ---------------- COMPUTE THE COMPONENTS OF THE RADIATIVE TENSOR ----------------------
    print('computing the components of the radiative tensor')

    Jm00 = integ.simps(phy_shape*II_new, mus)
    Jm00 = 1/2 * integ.simps( Jm00, wnorm)
    Jm02 = phy_shape * (3*mu_shape**2 - 1)*II_new + 3*(mu_shape**2 - 1)*QQ_new
    Jm02 = integ.simps( Jm02, mus )
    Jm02 = 1/np.sqrt(4**2 * 2) * integ.simps( Jm02, wnorm)

    # print('The Jm00 component is: {} and the Jm02 is: {}'.format(Jm00[-1],Jm02[-1]))
    # ---------------- COMPUTE THE SOURCE FUNCTIONS TO SOLVE THE RTE -----------------------
    print('Computing the source function to close the loop and solve the ETR again')
    
    # computing Jm00 and Jm02 with tensor shape as the rest of the variables
    Jm00_shape = np.repeat(np.repeat(Jm00[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)
    Jm02_shape = np.repeat(np.repeat(Jm02[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)

    # plt.plot(zz,(Jm00_shape/plank_Ishape)[:,125,-1], 'b--', label=r'$J^0_0/B_\nu$ shape')
    # plt.plot(zz,(Jm02_shape/plank_Ishape)[:,125,-1], 'r-.', label=r'$J^2_0/B_\nu$ shape')
    # plt.legend(); plt.show()

    # plt.imshow((Jm00_shape/plank_Ishape)[:,:,-1], origin='lower', aspect='equal'); plt.title(r'$J^0_0/B_\nu$');plt.colorbar(); plt.show()
    # plt.imshow((Jm02_shape/plank_Ishape)[:,:,-1], origin='lower', aspect='equal'); plt.title(r'$J^2_0/B_\nu$');plt.colorbar(); plt.show()

    S00 = (1-pm.eps)*Jm00_shape + pm.eps*plank_Ishape
    S20 = pm.Hd * (1-pm.eps)/(1 + (1-pm.eps)*pm.dep_col**2) * w2jujl * Jm02_shape

    SLI_new = S00 + w2jujl * (3*mu_shape**2 - 1)/np.sqrt(8) * S20
    SLQ_new = w2jujl * 3*(mu_shape**2 - 1)/np.sqrt(8) * S20

    SI_new = phy_shape/(phy_shape + pm.r)*SLI_new + pm.r/(phy_shape + pm.r)*plank_Ishape
    SQ_new = phy_shape/(phy_shape + pm.r)*SLQ_new

    # plt.plot(ww, SI[13,:,49], 'k', label=r'$S_I(\nu,z= ,\mu=1)$')
    # plt.plot(ww, SQ[13,:,49], 'k--', label=r'$S_Q(\nu,z= ,\mu=1)$')
    # plt.plot(ww, SLI[13,:,49], 'b', label=r'$S^L_I(\nu,z= ,\mu=1)$')
    # plt.plot(ww, SLQ[13,:,49], 'b--', label=r'$S^L_Q(\nu,z= ,\mu=1)$')
    # plt.plot(ww, SI_new[13,:,49], 'r', label=r'$S_{I,new}(\nu,z= ,\mu=1)$')
    # plt.plot(ww, SQ_new[13,:,49], 'r--', label=r'$S_{Q,new}(\nu,z= ,\mu=1)$')
    # plt.plot(ww, SLI_new[13,:,49], 'g', label=r'$S^L_{I,new}(\nu,z= ,\mu=1)$')
    # plt.plot(ww, SLQ_new[13,:,49], 'g--', label=r'$S^L_{Q,new}(\nu,z= ,\mu=1)$')
    # plt.legend()    
    # plt.show()

    # plt.imshow(SI[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S_I$');plt.colorbar(); plt.show()
    # plt.imshow(SI_new[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S_{I,new}$');plt.colorbar(); plt.show()
    # plt.imshow(SQ[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S_Q$');plt.colorbar(); plt.show()
    # plt.imshow(SQ_new[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S_{Q,new}$');plt.colorbar(); plt.show()
    # plt.imshow(SLI[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S^L_I$');plt.colorbar(); plt.show()
    # plt.imshow(SLI_new[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S^L_{I,new}$');plt.colorbar(); plt.show()
    # plt.imshow(SLQ[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S^L_Q$');plt.colorbar(); plt.show()
    # plt.imshow(SLQ_new[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S^L_{Q,new}$');plt.colorbar(); plt.show()
    # plt.imshow(S20[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S^2_0$');plt.colorbar(); plt.show()
    # plt.imshow(Jm02_shape[:,:,49], origin='lower', aspect='equal'); plt.title(r'$J^2_0$');plt.colorbar(); plt.show()



    print('Computing the differences and reasign the intensities')
    diff = np.append(np.append(np.append(II - II_new, QQ - QQ_new), SI-SI_new), SQ-SQ_new)
    II = np.copy(II_new)
    QQ = np.copy(QQ_new)
    SI = np.copy(SI_new)
    SQ = np.copy(SQ_new)
    SLI = np.copy(SLI_new)
    SLQ = np.copy(SLQ_new)

    if( np.all( np.abs(diff) < pm.tolerance ) ):
        print('-------------- FINISHED!!---------------')
        break

if (i >= pm.max_iter - 1):
    print('Ops! The solution with the desired tolerance has not been found')
    print('Although an aproximate solution may have been found. Try to change')
    print('the parameters to obtain an optimal solution.')
    print('The found tolerance is: ',np.max(diff))


plt.plot(ww, II[-1,:,50], color='k', label='I')
plt.plot(ww, QQ[-1,:,50], color='g', label='Q')
plt.plot(ww, SI[-1,:,50], color='b', label='Source function')
plt.plot(ww, SLI[-1,:,50], color='r', label='line source function')
plt.legend()
plt.show()
plt.plot(zz,II[:,40,-1], color='k', label='$I(z)$')
plt.plot(zz,QQ[:,40,-1], color='g', label='$Q(z)$')
plt.plot(zz,II_new[:,40,-1], color='b', label='$I_{new}(z)$')
plt.plot(zz,SI[:,40,-1], color = 'r', label = '$S_I$')
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
