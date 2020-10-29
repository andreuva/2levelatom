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
        if mus[j] < 0:
            for i in range(len(zz)-2,0,-1):
                psim,psio,psip = psi_calc(deltaum = deltau[i], deltaup = deltau[i-1])
                I_new[i,:,j] = I_new[i+1,:,j]*np.exp(-deltau[i]) + SI[i+1,:,j]*psim + SI[i,:,j]*psio + SI[i-1,:,j]*psip
                Q_new[i,:,j] = Q_new[i+1,:,j]*np.exp(-deltau[i]) + SQ[i+1,:,j]*psim + SQ[i,:,j]*psio + SQ[i-1,:,j]*psip

            psim, psio = psi_calc(deltaum = deltau[1], deltaup = deltau[0], mode='linear')
            I_new[0,:,j] = I_new[1,:,j]*np.exp(-deltau[0]) + SI[1,:,j]*psim + SI[0,:,j]*psio 
            Q_new[0,:,j] = Q_new[1,:,j]*np.exp(-deltau[0]) + SQ[1,:,j]*psim + SQ[0,:,j]*psio
        else:
            for i in range(1,len(zz)-1,1):
                psim,psio,psip = psi_calc(deltau[i-1], deltau[i])
                I_new[i,:,j] = I_new[i-1,:,j]*np.exp(-deltau[i-1]) + SI[i-1,:,j]*psim + SI[i,:,j]*psio + SI[i+1,:,j]*psip
                Q_new[i,:,j] = Q_new[i-1,:,j]*np.exp(-deltau[i-1]) + SQ[i-1,:,j]*psim + SQ[i,:,j]*psio + SQ[i+1,:,j]*psip
                
            psim, psio = psi_calc(deltau[-2], deltau[-1], mode='linear')
            I_new[-1,:,j] = I_new[-2,:,j]*np.exp(-deltau[-1]) + SI[-2,:,j]*psim + SI[-1,:,j]*psio 
            Q_new[-1,:,j] = Q_new[-2,:,j]*np.exp(-deltau[-1]) + SQ[-2,:,j]*psim + SQ[-1,:,j]*psio
    return I_new,Q_new

# -----------------------------------------------------------------------------------
# ---------------------- MAIN LOOP TO OBTAIN THE SOLUTION ---------------------------
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    # Compute the source function as a tensor in of zz, ww, mus
    # Initialaice the used tensors
    II = np.copy(plank_Ishape)
    II[1:] = II[1:]*0
    QQ = np.zeros_like(II)
    II_new = np.copy(II)
    QQ_new = np.zeros_like(QQ)


    S00 = np.copy(plank_Ishape)
    SLI = np.copy(S00)
    SLQ = np.zeros_like(SLI)
    SI = phy_shape/(phy_shape + pm.r) * SLI + (pm.r/(phy_shape + pm.r)) * plank_Ishape
    SQ = np.zeros_like(SI)                                           # SQ = 0 (size of SI)

    # plt.plot(ww, II[50,:,-1], color='k', label=r'$B_{\nu}(T= $'+'{}'.format(pm.T) + '$)$')
    # plt.plot(ww, QQ[50,:,-1], color='g', label=r'$Q(\nu,z=0,\mu=1)$')
    # plt.plot(ww, SI[50,:,-1], color='b', label=r'$S_I(\nu,z=0,\mu=1)$')
    # plt.plot(ww, SLI[50,:,-1], color='r', label=r'$S^L_I(\nu,z=0,\mu=1)$')
    # plt.xlabel(r'$\nu\ (Hz)$')
    # plt.legend()
    # plt.show()
    # plt.plot(ww, (phy/(phy + pm.r)), color='r', label= r'$ \dfrac{\phi(\nu)}{\phi(\nu) + r}$')
    # plt.plot(ww, pm.r/(phy + pm.r), color='b', label=r'$ \dfrac{r}{\phi(\nu) + r}$')
    # plt.plot(ww, phy_shape[0,:,0], color='k', label=r'$ \phi(\nu) $')
    # plt.xlabel(r'$\nu\ (Hz)$'); plt.title('profiles with $a=${} and $w_0=${:.3e} Hz'.format(pm.a,pm.w0))
    # plt.legend()
    # plt.show()
    w2jujl = jsymbols.j6(1,1,2,1,1,0)/jsymbols.j6(1,1,0,1,1,0)
    D2 = False
    plots = False

    for i in range(pm.max_iter):

        # ----------------- SOLVE RTE BY THE SHORT CHARACTERISTICS ---------------------------
        print('Solving the Radiative Transpor Equations')
        II_new, QQ_new = RTE_SC_solve(II,QQ,SI,SQ,zz,mus, 'imp')
        
        if np.min(II_new) < 0:
            print(np.unravel_index(np.argmin(II_new), II_new.shape))
            print(np.min(II_new))
            # exit()

        # ---------------- COMPUTE THE COMPONENTS OF THE RADIATIVE TENSOR ----------------------
        print('computing the components of the radiative tensor')

        Jm00 = integ.simps( phy_shape*II_new, wnorm, axis=1)
        Jm00 = 1/2 * integ.simps(Jm00, mus)
        Jm02 = phy_shape * (3*mu_shape**2 - 1)*II_new + 3*(mu_shape**2 - 1)*QQ_new
        Jm02 = integ.simps( Jm02, mus )
        Jm02 = 1/np.sqrt(4**2 * 2) * integ.simps( Jm02, wnorm)

        Jm00_shape = np.repeat(np.repeat(Jm00[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)
        Jm02_shape = np.repeat(np.repeat(Jm02[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)

        # ---------------- COMPUTE THE SOURCE FUNCTIONS TO SOLVE THE RTE -----------------------
        print('Computing the source function to close the loop and solve the ETR again')
        # computing Jm00 and Jm02 with tensor shape as the rest of the variables

        S00 = (1-pm.eps)*Jm00_shape + pm.eps*plank_Ishape
        S20 = pm.Hd * (1-pm.eps)/(1 + (1-pm.eps)*pm.dep_col**2) * w2jujl * Jm02_shape

        SLI_new = S00 + w2jujl * (3*mu_shape**2 - 1)/np.sqrt(8) * S20
        SLQ_new = w2jujl * 3*(mu_shape**2 - 1)/np.sqrt(8) * S20

        SI_new = phy_shape/(phy_shape + pm.r)*SLI_new + pm.r/(phy_shape + pm.r)*plank_Ishape
        SQ_new = phy_shape/(phy_shape + pm.r)*SLQ_new

        if plots:
            plt.plot(ww, (II/plank_Ishape)[-1, :, -1], 'b', label='$I$')
            plt.plot(ww, (II_new/plank_Ishape)[-1, :, -1], 'b--', label='$I_{calc}$')
            # plt.plot(ww, (QQ/plank_Ishape)[-1, :, -1], 'r', label='$Q$')
            # plt.plot(ww, (QQ_new/plank_Ishape)[-1, :, -1], 'r--', label='$Q_{calc}$')
            plt.plot(ww, (SI/plank_Ishape)[-1,:,-1], 'm', label=r'$S_I$')
            plt.plot(ww, (SI_new/plank_Ishape)[-1,:,-1], 'm--', label=r'$S_{I,new}$')            
            plt.plot(ww, (SLI/plank_Ishape)[-1,:,-1], 'g', label=r'$S^L_I$')
            plt.plot(ww, (SLI_new/plank_Ishape)[-1,:,-1], 'g--', label=r'$S^L_{I,new}$')
            # plt.plot(ww, (SQ/plank_Ishape)[-1,:,-1], 'k', label=r'$S_Q$')
            # plt.plot(ww, (SQ_new/plank_Ishape)[-1,:,-1], 'k--', label=r'$S_{Q,new}$')
            # plt.plot(ww, (SLQ/plank_Ishape)[-1,:,-1], 'k:', label=r'$S^L_Q$')
            # plt.plot(ww, (SLQ_new/plank_Ishape)[-1,:,-1], 'k-.', label=r'$S^L_{Q,new}$')
            plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
            plt.show()
            plt.plot(zz, (II_new/plank_Ishape)[:, 50, -1], 'b', label='$I$')
            plt.plot(zz, (QQ_new/plank_Ishape)[:, 50, -1], 'b--', label='$Q$')
            plt.plot(zz,(Jm00_shape/plank_Ishape)[:,50,-1], 'g', label=r'$J^0_0/B_\nu$ shape')
            plt.plot(zz,(Jm02_shape/plank_Ishape)[:,50,-1], 'g--', label=r'$J^2_0/B_\nu$ shape')
            plt.legend(); plt.xlabel('z'); plt.show()
            plt.plot(zz, (II_new/plank_Ishape)[:, 50, 1], 'b', label='$I$')
            plt.plot(zz, (QQ_new/plank_Ishape)[:, 50, 1], 'b--', label='$Q$')
            plt.plot(zz,(Jm00_shape/plank_Ishape)[:,50,1], 'g', label=r'$J^0_0/B_\nu$ shape')
            plt.plot(zz,(Jm02_shape/plank_Ishape)[:,50,1], 'g--', label=r'$J^2_0/B_\nu$ shape')
            plt.legend(); plt.xlabel('z'); plt.show()
        if D2:
            plt.imshow(II[:, :, -1], origin='lower', aspect='equal'); plt.title('$I$'); plt.colorbar(); plt.show()
            plt.imshow(II_new[:, :, -1], origin='lower', aspect='equal'); plt.title('$I_{calc}$');plt.colorbar(); plt.show()
            plt.imshow(QQ[:, :, -1], origin='lower', aspect='equal'); plt.title('$Q$'); plt.colorbar(); plt.show()
            plt.imshow(QQ_new[:, :, -1], origin='lower', aspect='equal'); plt.title('$Q_{calc}$');plt.colorbar(); plt.show()
            plt.imshow(SI[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S_I$');plt.colorbar(); plt.show()
            plt.imshow(SI_new[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S_{I,new}$');plt.colorbar(); plt.show()
            # plt.imshow(SQ[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S_Q$');plt.colorbar(); plt.show()
            plt.imshow(SQ_new[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S_{Q,new}$');plt.colorbar(); plt.show()
            # plt.imshow(SLI[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S^L_I$');plt.colorbar(); plt.show()
            plt.imshow(SLI_new[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S^L_{I,new}$');plt.colorbar(); plt.show()
            # plt.imshow(SLQ[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S^L_Q$');plt.colorbar(); plt.show()
            plt.imshow(SLQ_new[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S^L_{Q,new}$');plt.colorbar(); plt.show()
            # plt.imshow(S20[:,:,49], origin='lower', aspect='equal'); plt.title(r'$S^2_0$');plt.colorbar(); plt.show()
            # plt.imshow(Jm02_shape[:,:,49], origin='lower', aspect='equal'); plt.title(r'$J^2_0$');plt.colorbar(); plt.show()
            plt.imshow((Jm00_shape/plank_Ishape)[:,:,-1], origin='lower', aspect='equal'); plt.title(r'$J^0_0/B_\nu$');plt.colorbar(); plt.show()
            plt.imshow((Jm02_shape/plank_Ishape)[:,:,-1], origin='lower', aspect='equal'); plt.title(r'$J^2_0/B_\nu$');plt.colorbar(); plt.show()


        print('Computing the differences and reasign the intensities')
        olds = np.array([II, QQ, SI, SQ])
        news = np.array([II_new, QQ_new, SI_new, SQ_new])
        II = np.copy(II_new)
        QQ = np.copy(QQ_new)
        SI = np.copy(SI_new)
        SQ = np.copy(SQ_new)
        SLI = np.copy(SLI_new)
        SLQ = np.copy(SLQ_new)
        diff = np.abs(olds - news)
        tol = np.max(diff)/news[np.unravel_index(np.argmax(diff), diff.shape)]
        print('Actual tolerance is :',tol*100,'%')
        if( tol < pm.tolerance ):
            print('-------------- FINISHED!!---------------')
            break

    if (i >= pm.max_iter - 1):
        print('Ops! The solution with the desired tolerance has not been found')
        print('Although an aproximate solution may have been found. Try to change')
        print('the parameters to obtain an optimal solution.')
        print('The found tolerance is: ',tol*100, '%')

    plt.plot(ww, (II/plank_Ishape)[-1,:,-1], 'b', label='$I$')
    plt.plot(ww, (QQ/II)[-1,:,-1], 'r', label='$Q$')
    plt.plot(ww, (SI/plank_Ishape)[-1,:,-1], 'b--', label=r'$S_I$')
    plt.plot(ww, (SQ/SI)[-1,:,-1], 'r--', label=r'$S_Q$')
    plt.legend(); plt.xlabel(r'$\nu\ (Hz)$'); plt.show()
    plt.plot(zz, (II/plank_Ishape)[:, 50, -1], 'b', label='$I$')
    plt.plot(zz, (QQ/II)[:, 50, -1], 'b--', label='$Q$')
    plt.plot(zz,(Jm00_shape/plank_Ishape)[:,50,-1], 'g', label=r'$J^0_0/B_\nu$ shape')
    plt.plot(zz,(Jm02_shape/plank_Ishape)[:,50,-1], 'g--', label=r'$J^2_0/B_\nu$ shape')
    plt.plot(zz, (SI/plank_Ishape)[:, 50, -1], 'k', label='$S_I$')
    plt.plot(zz, (SQ/II)[:, 50, -1], 'k--', label='$S_Q$')
    plt.legend(); plt.xlabel('z'); plt.show()

    # tolerancia en todas las capas en SO0, S02

    # -------------------- ONCE WE OBTAIN THE SOLUTION, COMPUTE THE POPULATIONS ----------------
    Jm00_shape = np.repeat(np.repeat(Jm00[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)
    Jm02_shape = np.repeat(np.repeat(Jm02[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)

    rho00 = np.sqrt((2*pm.ju + 1)/(2*pm.jl+1)) * (2*cte.h*ww_shape**3/cte.c**2)**-1 * \
        ((1-pm.eps)*Jm00_shape + pm.eps*plank_Ishape)/((1-pm.eps)*pm.dep_col + 1)

    rho02 = np.sqrt((2*pm.ju + 1)/(2*pm.jl+1)) * (2*cte.h*ww_shape**3/cte.c**2)**-1 * \
        ((1-pm.eps)*Jm02_shape)/((1-pm.eps)*(1j*pm.Hd*2 + pm.dep_col) + 1)
