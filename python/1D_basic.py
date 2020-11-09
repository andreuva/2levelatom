#############################################################################
#                       1D NAIVE ATMOSPHERE SOLVER                          #
#                      AUTHOR: ANDRES VICENTE AREVALO                       #
#############################################################################
import matplotlib.pyplot as plt
import numpy as np

# Define the grid points in the z direction (with Xi = exp(-z))
zl = -15
zu = 8
dz = .1

nz = int((zu-zl)/dz)
zz = np.arange(zl,zu+dz,dz)

#Define the maximum tolerance and the maximum iterations posible
max_iter = 5000
tolerance = 1e-6

# Phot. dest. probability (LTE=1,NLTE=1e-4)
eps = 1e-4

# Define the 2 directions of the gaussian cuadrature (qnd=2 up and down)
mu_up = 1/np.sqrt(3)
mu_down = -1/np.sqrt(3)

# Define the source function as LTE (SI = 1)
SI = np.ones((len(zz)))

# Put the intensities at 0 and define the boundary conditions 
# I_up[0] = 1 I_down[-1] = 0
II_up = np.zeros((len(zz)))
II_down = np.zeros((len(zz)))
II_up[0] = 1

#define the new intensities as a copy of the previous
SI_new = SI.copy()

# compute the tau and define the initial lambda functions (0)
tau = np.exp(-zz)
lmb_up = np.zeros_like(II_up)
lmb_down = np.zeros_like(II_down)

# Compute the analitic solution
SI_analitic = (1-eps)*(1-np.exp(-tau*np.sqrt(3*eps))/(1+np.sqrt(eps))) + eps

# initialice the vectors to hold the MRC and the absolute error in each iteration
mrc = []
error = []


def psicalc(deltaum, deltaup, mode=1):
    """
    Compute of the psi coefficients in the SC method
    """
    U0 = 1 - np.exp(-deltaum)
    U1 = deltaum - U0

    if mode == 1:
        psim = U0 - U1/deltaum
        psio = U1/deltaum
        return psim, psio
    else:
        U2 = (deltaum)**2 - 2*U1

        psim = U0 + (U2 - U1*(deltaup + 2*deltaum))/(deltaum*(deltaum + deltaup))
        psio = (U1*(deltaum + deltaup) - U2)/(deltaum*deltaup)
        psip = (U2 - U1*deltaum)/(deltaup*(deltaup+deltaum))
        return psim, psio, psip


def RT_1D(I_up, I_down, SI, l_u, l_d, tau, mu_u, mu_d):
    """
    Compute the new intensities form the source function with the SC method
    """

    psip_prev = 0
    for i in range(1,len(tau)):
        deltaum = np.abs((tau[i-1] - tau[i])/mu_u)

        if (i < (len(tau)-1)):
            deltaup = np.abs((tau[i] - tau[i+1])/mu_u)
            psim, psio, psip = psicalc(deltaum, deltaup, mode = 2)
            I_up[i] = I_up[i-1]*np.exp(-deltaum) + psim*SI[i-1] + psio*SI[i] + psip*SI[i+1]
            lmb_up[i] = psip_prev*np.exp(-deltaum) + psio
            psip_prev = psip
        else:
            psim, psio = psicalc(deltaum, deltaum, mode = 1)
            I_up[i] = I_up[i-1]*np.exp(-deltaum) + psim*SI[i-1] + psio*SI[i]
            lmb_up[i] = psip_prev*np.exp(-deltaum) + psio

    psip_prev = 0
    for i in range(len(tau)-2,-1,-1):
        deltaum = np.abs((tau[i] - tau[i+1])/mu_d)

        if (i > 0):
            deltaup = np.abs((tau[i-1] - tau[i])/mu_d)

            psim, psio, psip = psicalc(deltaum, deltaup, mode = 2)
            I_down[i] = I_down[i+1]*np.exp(-deltaum) + psim*SI[i+1] + psio*SI[i] + psip*SI[i-1]
            lmb_down[i] = psip_prev*np.exp(-deltaum) + psio
            psip_prev = psip
        else:
            psim, psio = psicalc(deltaum, deltaum, mode = 1)
            I_down[i] = I_down[i+1]*np.exp(-deltaum) + psim*SI[i+1] + psio*SI[i]
            lmb_down[i] = psip_prev*np.exp(-deltaum) + psio

    return I_up, I_down, lmb_up, lmb_down

# --------------------------------------------------------------------------------- #
# ---------------------- MAIN LOOP TO OBTAIN THE SOLUTION ------------------------- #
# --------------------------------------------------------------------------------- #
it = 1; tol = 1
while(it < max_iter and tol > tolerance):
    
    # Compute the new II and lambda
    II_up, II_down, lmb_up, lmb_down = RT_1D(II_up, II_down, SI, lmb_up, lmb_down, tau, mu_up, mu_down)
    # Compute the J and lambda integrating both directions
    J = 1/2 * (II_up + II_down)
    lmb_integ = 1/2 * (lmb_up + lmb_down)

    # Compute the new source function with the integrated quantities
    SI_new = (1-eps)*J + eps
    SI_new = (SI_new - SI)/(1 - (1-eps)*lmb_integ) + SI

    # Compute the MRC and the error and storing it
    tol = np.max(np.abs(np.abs(SI - SI_new)/(SI+1e-200)))
    err = np.max(np.abs(SI-SI_analitic))
    mrc.append(tol)
    error.append(err)

    # Copying the SI to the old vector and counting the iterations
    SI = SI_new.copy()
    it = it + 1


# PLOT THE SOLUTIONS AND THE ERRORS

print(f'Desired tolerance was {tolerance} with a maximum iterations: {max_iter}')
print(f'Finished after {it} iterations with a tolerance of: {tol}')
plt.title('Solution of the atmosphere with 2 directions')
plt.xlabel('zz'); plt.ylabel(r'$I/B_{\nu}$')
plt.plot(zz,II_down, 'b--', label = '$I_{down}$')
plt.plot(zz,II_up, 'r--', label = '$I_{up}$')
plt.plot(zz,SI, 'k', label='$S_I$ solution')
plt.plot(zz,SI_analitic, 'pink', label = '$S_I$ analitic')
plt.legend()
plt.show()

plt.title('MCR & ERROR')
plt.xlabel('itt'); plt.ylabel(r'$log_{10}(MRC)$ & $log_{10}(error)$')
plt.plot(np.log10(error), 'b--', label = 'error (SI-SI_analitic)')
plt.plot(np.log10(mrc), 'b-',label='MRC')
plt.xscale('log')
plt.legend()
plt.show()