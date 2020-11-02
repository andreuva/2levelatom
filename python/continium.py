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

# Initialaice the intensities vectors to solve the ETR
# Computed as a tensor in zz, ww, mus
plank_Ishape = np.repeat(np.repeat(func.plank_wien(ww, pm.T)[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
mu_shape = np.repeat(np.repeat(mus[np.newaxis,:], len(ww), axis=0)[np.newaxis, :, :], len(zz), axis=0)
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

    n=50
    plots = False

    # Compute the source function as a tensor in of zz, ww, mus
    # Initialaice the used tensors
    II = np.copy(plank_Ishape)
    II[1:] = plank_Ishape[1:]*0
    QQ = np.zeros_like(II)
    II_new = np.copy(II)
    QQ_new = np.zeros_like(QQ)

    SI = np.copy(plank_Ishape)
    SQ = np.zeros_like(SI)                                           # SQ = 0 (size of SI)

    for i in range(pm.max_iter):

        # ----------------- SOLVE RTE BY THE SHORT CHARACTERISTICS ---------------------------
        print('Solving the Radiative Transpor Equations')
        II_new, QQ_new = RTE_SC_solve(II,QQ,SI,SQ,zz,mus, 'imp')
        
        if np.min(II_new) < 0:
            print(np.unravel_index(np.argmin(II_new), II_new.shape))
            print(np.min(II_new))
            break

        if plots:
            plt.plot(zz, (II/plank_Ishape)[:, n, -1], 'b', label='$I$')
            plt.plot(zz, (QQ/plank_Ishape)[:, n, -1], 'r--', label='$Q$')
            plt.plot(zz, (II_new/plank_Ishape)[:, n, -1], 'b-.', label='$I_{calc}$')
            plt.plot(zz, (QQ_new/plank_Ishape)[:, n, -1], 'r.', label='$Q_{calc}$')
            plt.legend(); plt.xlabel('z')
            plt.show()

        # ---------------- COMPUTE THE COMPONENTS OF THE RADIATIVE TENSOR ----------------------
        print('computing the components of the radiative tensor')

        Jm00 = 1/2 * integ.simps(II_new, mus )
        Jm02 = 1/np.sqrt(4**2 * 2) * integ.simps( (3*mu_shape**2 - 1)*II_new + 3*(mu_shape**2 - 1)*QQ_new, mus )

        Jm00_shape = np.repeat(Jm00[ :, :, np.newaxis], len(mus), axis=2)
        Jm02_shape = np.repeat(Jm02[ :, :, np.newaxis], len(mus), axis=2)

        # ---------------- COMPUTE THE SOURCE FUNCTIONS TO SOLVE THE RTE -----------------------
        print('Computing the source function to close the loop and solve the ETR again')
        # computing Jm00 and Jm02 with tensor shape as the rest of the variables

        if plots:
            plt.plot(zz,(Jm00_shape/plank_Ishape)[:,n,-1], 'b--', label=r'$J^0_0/B_\nu$')
            plt.plot(zz,(Jm02_shape/plank_Ishape)[:,n,-1], 'r-.', label=r'$J^2_0/B_\nu$')
            plt.legend(); plt.show()
            plt.plot(mus,(Jm00_shape/plank_Ishape)[1,n,:], 'b--', label=r'$J^0_0/B_\nu$')
            plt.plot(mus,(Jm02_shape/plank_Ishape)[1,n,:], 'r-.', label=r'$J^2_0/B_\nu$')
            plt.legend(); plt.show()

        SI_new = (1-pm.eps)*Jm00_shape + pm.eps*plank_Ishape
        SQ_new = (1-pm.eps)*Jm02_shape

        print('Computing the differences and reasign the intensities')
        olds = np.append(np.append(np.append(II, QQ), SI), SQ)
        news = np.append(np.append(np.append(II_new, QQ_new), SI_new), SQ_new)
        II = np.copy(II_new)
        QQ = np.copy(QQ_new)
        SI = np.copy(SI_new)
        SQ = np.copy(SQ_new)
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
        print('The found tolerance is: ',tol*100)
    
    print('finished after :',i,' iterations')

    plt.imshow((II)[:, n, :], origin='lower', aspect='equal'); plt.xlabel('$\mu$'); plt.ylabel('z'); plt.title('$I$'); plt.colorbar(); plt.show()
    plt.imshow((QQ)[:, n, :], origin='lower', aspect='equal'); plt.xlabel('$\mu$'); plt.ylabel('z'); plt.title('$Q$'); plt.colorbar(); plt.show()
    plt.imshow((SI)[:,n,:], origin='lower', aspect='equal'); plt.xlabel('$\mu$'); plt.ylabel('z'); plt.title(r'$S_I$');plt.colorbar(); plt.show()
    plt.imshow((SQ)[:,n,:], origin='lower', aspect='equal'); plt.xlabel('$\mu$'); plt.ylabel('z'); plt.title(r'$S_Q$');plt.colorbar(); plt.show()
    plt.imshow((Jm00_shape/plank_Ishape)[:,n,:], origin='lower', aspect='equal'); plt.xlabel('$\mu$'); plt.ylabel('z'); plt.title(r'$J^0_0/B_\nu$');plt.colorbar(); plt.show()
    plt.imshow((Jm02_shape/plank_Ishape)[:,n,:], origin='lower', aspect='equal'); plt.xlabel('$\mu$'); plt.ylabel('z'); plt.title(r'$J^2_0/B_\nu$');plt.colorbar(); plt.show()


    plt.plot(zz,(II/plank_Ishape)[:,n,-1], 'k', label=r'$I/B_{\nu}$')
    plt.plot(zz,(QQ)[:,n,-1], 'g', label=r'$Q/I$')
    plt.plot(zz,(SI/plank_Ishape)[:,n,-1], 'k--', label = r'$S_I/B_{\nu}$')
    plt.plot(zz,(SQ/SI)[:,n,-1], 'g--', label = r'$S_Q/S_I$')
    plt.plot(zz,(Jm00_shape/plank_Ishape)[:,n,-1], 'r', label=r'$J^0_0/B_\nu$ shape')
    plt.plot(zz,(Jm02_shape/plank_Ishape)[:,n,-1], 'r--', label=r'$J^2_0/B_\nu$ shape')
    plt.legend()
    plt.show()

    # tolerancia en todas las capas en SO0, S02
    '''
    # -------------------- ONCE WE OBTAIN THE SOLUTION, COMPUTE THE POPULATIONS ----------------
    Jm00_shape = np.repeat(np.repeat(Jm00[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)
    Jm02_shape = np.repeat(np.repeat(Jm02[ :, np.newaxis], len(ww), axis=1)[ :, :, np.newaxis], len(mus), axis=2)

    rho00 = np.sqrt((2*pm.ju + 1)/(2*pm.jl+1)) * (2*cte.h*ww_shape**3/cte.c**2)**-1 * \
        ((1-pm.eps)*Jm00_shape + pm.eps*plank_Ishape)/((1-pm.eps)*pm.dep_col + 1)

    rho02 = np.sqrt((2*pm.ju + 1)/(2*pm.jl+1)) * (2*cte.h*ww_shape**3/cte.c**2)**-1 * \
        ((1-pm.eps)*Jm02_shape)/((1-pm.eps)*(1j*pm.Hd*2 + pm.dep_col) + 1)'''
