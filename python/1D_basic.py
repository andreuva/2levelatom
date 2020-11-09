#############################################################################
#                       1D NAIVE ATMOSPHERE SOLVER                          #
#                      AUTHOR: ANDRES VICENTE AREVALO                       #
#############################################################################
import matplotlib.pyplot as plt
import numpy as np

max_iter = 1e5
tolerance = 1e-8

zl = -15
zu = 8
dz = 1

nz = int((zu-zl)/dz)
zz = np.arange(zl,zu+dz,dz)

eps = 1e-4

mu_up = 1
mu_down = -1

SI = np.ones((len(zz)))
II_up = np.zeros((len(zz)))
II_down = np.zeros((len(zz)))
II_up[0] = 1

II_new_up = II_up.copy()
II_new_down = II_down.copy()
SI_new = SI.copy()

tau = np.exp(-zz)
lmb_up = np.zeros_like(II_up)
lmb_down = np.zeros_like(II_down)

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


def RT_1D(I_u, I_d, SI, l_u, l_d, tau):
    I_up = I_u.copy()
    I_down = I_d.copy()
    lmb_up = l_d.copy()
    lmb_down = l_u.copy()

    psip_prev = 0
    for i in range(1,len(tau)):
        deltaum = (tau[i-1] - tau[i])

        if (i < (len(tau)-1)):
            deltaup = (tau[i] - tau[i+1])

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
        deltaum = (tau[i] - tau[i+1])

        if (i > 0):
            deltaup = (tau[i-1] - tau[i])

            psim, psio, psip = psicalc(deltaum, deltaup, mode = 2)
            I_down[i] = I_down[i+1]*np.exp(-deltaum) + psim*SI[i+1] + psio*SI[i] + psip*SI[i-1]
            lmb_down[i] = psip_prev*np.exp(-deltaum) + psio
            psip_prev = psip
        else:
            psim, psio = psicalc(deltaum, deltaum, mode = 1)
            I_down[i] = I_down[i+1]*np.exp(-deltaum) + psim*SI[i+1] + psio*SI[i]
            lmb_down[i] = psip_prev*np.exp(-deltaum) + psio

    return I_up, I_down, lmb_up, lmb_down

tol = 1
mrc = []
error = []
it = 0

SI_analitic = (1-eps)*(1-np.exp(-tau*np.sqrt(3*eps))/(1+np.sqrt(eps))) + eps

lmb_integ = 0
while(it < max_iter and tol > tolerance):
    
    II_new_up, II_new_down, lmb_up, lmb_down = RT_1D(II_up, II_down, SI, lmb_up, lmb_down, tau)
    J = 1/2 * (II_new_up + II_new_down)
    lmb_integ = 1/2 * (lmb_up + lmb_down)
    SI_new = (1-eps)*J + eps

    SI_new = (SI_new - SI)/(1 - (1-eps)*lmb_integ) + SI

    tol = np.max(np.abs(np.abs(SI - SI_new)/(SI+1e-200)))
    err = np.max(np.abs(SI-SI_analitic))

    mrc.append(tol)
    error.append(err)
    II_down = II_new_down.copy()
    II_up = II_new_up.copy()
    SI = SI_new.copy()
    it = it + 1

print(it)
plt.plot(SI_analitic, label = 'analitic solution')
plt.plot(SI, label='found solution')
plt.plot(II_down, label = 'I down')
plt.plot(II_up, label = 'I up')
plt.legend()
plt.show()

plt.plot(np.log10(error))
plt.plot(np.log10(mrc))
plt.xscale('log')
plt.show()