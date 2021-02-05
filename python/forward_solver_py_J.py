#############################################################################
#                2 LEVEL ATOM ATMOSPHERE SOLVER                             #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm,trange
import parameters as pm
import physical_functions as func
import gaussian_quadrature as gauss
# from jsymbols import jsymbols
# jsymbols = jsymbols()

#############################################################################
#                   SUBROUTINES TO SOLVE THE PROBLEM                        #
#############################################################################
def psi_calc(deltaum, deltaup, mode='quadratic'):
    """
    Compute of the psi coefficients in the SC method given the deltatau's
    and the mode of the SI suposition: 'quadratic' or 'lineal'
    """
    U0 = 1 - np.exp(-deltaum)
    to_taylor = deltaum < 1e-3
    U0[to_taylor] = deltaum[to_taylor] - deltaum[to_taylor]**2/2 + deltaum[to_taylor]**3/6
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
    with a specified directions (mus) and optical depths (tau) wich are an 
    array of length nw (dimension of frequency's) and nz x nw matrix (as the
    tau depends on the height and the frequency)
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


def trapezoidal(y,x, axis=-1):
    """
    Function to integrate a array (1D or 2D in the last dimension) with the
    cumulative trapezoidal method given the x cordinates and the y values at those points.
    """
    h = x[1:] - x[:-1]

    if axis==0:
        I = np.sum((y[:-1] + y[1:])*h/2)
    else:
        I = np.zeros_like(y[:,0])
        for i in range(y.shape[0]):
            I[i] = np.sum((y[i,:-1] + y[i,1:])*h/2)

    return I

#############################################################################
#                   COMPUTATION OF THE PROFILES                             #
#                   AUTHOR: ANDRES VICENTE AREVALO                          #
#############################################################################
def solve_profiles( a, r, eps, dep_col, Hd):
    """
    Forward solver to obtain the I and Q profiles from the parameters of
    the atmosphere:
    a : Voigt profile dumping parameter
    r : line strength parameter for the integrated opacity X^C_I/X^L_I
    eps: photon destruction probability (eps = Clu/(Aul+Cul))
    dep_col: depolarization colision rate (delta = D^k_u/Aul)
    Hd: Hanle depolarization factor
    """

    # We define the z0, zl, dz as our heigt grid (just 1D because of a
    # plane-parallel atmosfere and axial-simetry)
    zz = np.arange(pm.zl, pm.zu + pm.dz, pm.dz)          # compute the 1D grid

    # Define the grid in frequencies ( or wavelengths )

    ww = np.arange(pm.wl, pm.wu + pm.dw, pm.dw)          # compute the 1D grid
    wnorm = ww.copy()

    # Define the directions of the rays
    if pm.qnd%2 != 0:
        pm.qnd = pm.qnd + 1
        print('To avoid singularity at mu=0' + f' the points of the cuadrature has change from {pm.qnd-1} to {pm.qnd}')

    [weigths, mus, err] = gauss.GaussLegendreWeights(pm.qnd, 1e-10)
    weigths_shape = np.repeat(np.repeat(weigths[np.newaxis,:], len(wnorm), axis=0)[np.newaxis, :, :], len(zz), axis=0)

    # ------------------------ SOME INITIAL CONDITIONS --------------------------
    # Compute the initial Voigts vectors
    phy = np.zeros_like(wnorm)
    for i in range(len(wnorm)):
        phy[i] = np.real(func.voigt(wnorm[i], a))
    phy = phy/trapezoidal(phy, wnorm, 0)          # normalice phy to sum 1


    # Initialaice the intensities vectors to solve the ETR
    # Computed as a tensor in zz, ww, mus
    plank_Ishape = np.repeat(np.repeat(np.ones_like(wnorm)[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
    mu_shape = np.repeat(np.repeat(mus[np.newaxis,:], len(wnorm), axis=0)[np.newaxis, :, :], len(zz), axis=0)
    phy_shape = np.repeat(np.repeat(phy[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
    # wnorm_shape = np.repeat(np.repeat(wnorm[ :, np.newaxis], len(mus), axis=1)[np.newaxis, :, :], len(zz), axis=0)
    zz_shape = np.repeat(np.repeat(zz[ :, np.newaxis], len(wnorm), axis=1)[:, :, np.newaxis], len(mus), axis=2)
    tau_shape = np.exp(-zz_shape)*(phy_shape + r)


    w2jujl = 1 # jsymbols.j6(1,1,2,1,1,0)/jsymbols.j6(1,1,0,1,1,0)
    rr = phy_shape/(phy_shape + r)

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
    
    # -----------------------------------------------------------------------------------
    # ---------------------- MAIN LOOP TO OBTAIN THE SOLUTION ---------------------------
    # -----------------------------------------------------------------------------------
    # t = trange(pm.max_iter, desc='Iterations:', leave=True)
    for itt in range(pm.max_iter): #t
        # ----------------- SOLVE RTE BY THE SHORT CHARACTERISTICS ---------------------------
        II, QQ, lambd = RTE_SC_solve(II, QQ, SI, SQ, tau_shape[:,:,-1], mus)
        
        if np.min(II) < -1e-4:
            print('fs.py found a negative intensity, stopping')
            print('Bad parameters:')
            print(f" a = {a}\n r = {r}\n eps = {eps}\n delta = {dep_col}\n Hd = {Hd}\n")
            break

        # ---------------- COMPUTE THE COMPONENTS OF THE RADIATIVE TENSOR ----------------------

        Jm00 = 1/2. * np.sum(II*weigths_shape, axis=-1)
        Jm00 = trapezoidal(phy_shape[:,:,0]*Jm00, wnorm)

        Jm02 = (3*mu_shape**2 - 1)*II + 3*(mu_shape**2 - 1)*QQ
        Jm02 = 1/np.sqrt(4**2 * 2) * np.sum(Jm02*weigths_shape, axis=-1)
        Jm02 = trapezoidal( phy_shape[:,:,0]*Jm02, wnorm)

        lambd = 1/2. * np.sum(rr*lambd*weigths_shape, axis=-1)
        lambd = trapezoidal( phy_shape[:,:,0] * lambd, wnorm)
        
        Jm00_shape = np.repeat(np.repeat(Jm00[ :, np.newaxis], len(wnorm), axis=1)[ :, :, np.newaxis], len(mus), axis=2)
        Jm02_shape = np.repeat(np.repeat(Jm02[ :, np.newaxis], len(wnorm), axis=1)[ :, :, np.newaxis], len(mus), axis=2)
        lambd = np.repeat(np.repeat(lambd[ :, np.newaxis], len(wnorm), axis=1)[ :, :, np.newaxis], len(mus), axis=2)

        # ---------------- COMPUTE THE SOURCE FUNCTIONS TO SOLVE THE RTE -----------------------

        S00_new = (1-eps)*Jm00_shape + eps*plank_Ishape
        S20 = Hd * (1-eps)/(1 + (1-eps)*dep_col) * w2jujl * Jm02_shape
        S00_new = (S00_new - S00)/(1 - (1-eps)*lambd) + S00

        SLI = S00_new + w2jujl * (3*mu_shape**2 - 1)/np.sqrt(8) * S20
        SLQ = w2jujl * 3*(mu_shape**2 - 1)/np.sqrt(8) * S20

        SI_new = rr*SLI + (1 - rr)*plank_Ishape
        SQ_new = rr*SLQ

        olds = np.array([SI])
        news = np.array([SI_new])
        SI = SI_new.copy()
        S00 = S00_new.copy()
        SQ = SQ_new.copy()
        tol = np.max(np.abs(np.abs(olds - news)/(olds+1e-200)))

        # t.set_description('Actual tolerance is : %1.3e' % tol)
        # t.refresh() # to show immediately the update

        if( tol < pm.tolerance ):
            break

    return II,QQ,Jm00,Jm02