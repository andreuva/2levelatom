#############################################################################
#    INVERSION CODE EXAMPLE USING FORWARD_SOLVER MODEL ATMOSPHERE           #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
import forward_solver_py  as fs
import random
import scipy.optimize as opt
from scipy.optimize import Bounds
from ctypes import c_void_p ,c_double, c_int, c_float, cdll
from numpy.ctypeslib import ndpointer

##########################     PARAMETERS     ##############################
a_sol = 1e-8      #1e-5,1e-2 ,1e-4                # dumping Voigt profile a=gam/(2^1/2*sig)
r_sol = 1e-8      #1,1e-4   ,1e-10                 # XCI/XLI
eps_sol = 1e-4                          # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col_sol = 0             #0.1          # Depolirarization colisions (delta)
Hd_sol = 0.75                  #1          # Hanle depolarization factor [1/5, 1]

zl = -15
zu = 9
dz = 0.75
nz = int((zu-zl)/dz + 1)

wl = -10
wu = 10
dw = 0.25
nw = int((wu-wl)/dw + 1)

qnd = 14
##########################     SUBROUTINES     ##############################
lib = cdll.LoadLibrary("/home/andreuva/Documents/2 level atom/2levelatom/c_python_integration/forward_solver.so")
solve_profiles = lib.solve_profiles
solve_profiles.restype = ndpointer(dtype=c_double , shape=(nz*nw*qnd*2,))

def add_noise(array, sigma):
    """
    Add nromal noise to an N dimensional array with a ginven sigma
    """
    noise = np.random.normal(0,sigma,array.size)
    noise = noise.reshape(array.shape)
    array = array + noise
    return array


def chi2(params, std, w_I, w_Q):
    '''
    Compute the cost function of the inversion given the parameters, the noise
    and weigths of the diferent components (I,Q...)
    '''
    a, r, eps, dep_col, Hd = params

    # print("\n New parameters: ")
    # print(f" a = {a}\n r = {r}\n eps = {eps}\n delta = {dep_col}\n Hd = {Hd}\n")
    print("\n Computing the new profiles:")

    # result = solve_profiles(c_float(a), c_float(r), c_float(eps), c_float(dep_col), c_float(Hd))
    # I = result[:nz*nw*qnd].reshape(I_sol.shape)
    # Q = result[nz*nw*qnd:].reshape(Q_sol.shape)
    I,Q = fs.solve_profiles(a, r, eps, dep_col, Hd)
    chi2 = np.sum(w_I*(I-I_sol)**2/std**2 + w_Q*(Q-Q_sol)**2/std**2)/(2*I.size)

    print(f'Chi^2 of this profiles is: {np.max(np.abs(chi2))} ' )
    return chi2

####################    COMPUTE THE "OBSERVED" PROFILE    #####################
print(" Solution parameters: ")
print(f" a = {a_sol}\n r = {r_sol}\n eps = {eps_sol}\n delta = {dep_col_sol}\n Hd = {Hd_sol}\n")
print("Computing the solution profiles:")
I_sol, Q_sol = fs.solve_profiles(a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol)

if(np.min(I_sol) < 0):
    print('Bad solution parameters, stopping.')
    exit()

std = 2e-6
I_sol = add_noise(I_sol, std)
Q_sol = add_noise(Q_sol, std)

##############  INITIALICE THE PARAMETERS AND COMPUTE INITIAL PROFILES #########
w_I = 1
w_Q = 1e2

random.seed(12455)
a = random.uniform(1e-10,1)
r = random.uniform(1e-15,1)
eps = random.uniform(1e-4,1)
dep_col = random.uniform(0,10)
Hd = random.uniform(1/5, 1)
print("\nInitial parameters: ")
print(f" a = {a}\n r = {r}\n eps = {eps}\n delta = {dep_col}\n Hd = {Hd}\n")
print("Computing the initial profiles:")
I_initial, Q_initial = fs.solve_profiles(a, r, eps, dep_col, Hd)



##########  MINIMIZE THE CHI2 FUNCTION WITH GIVEN RANGE CONSTRAINS #########
x0 = np.array([a,r,eps,dep_col,Hd])
xl = [1e-4,1e-12,1e-4,0,0.2]
xu = [1,1,1,10,1]
bounds = Bounds(xl,xu)
result = opt.minimize(chi2,x0, bounds=bounds, args=(std, w_I, w_Q))


##### PRINT AND PLOT THE SOLUTION AND COMPARE IT TO THE INITIAL AND OBSERVED PROFILES ####
print('\nFound Parameters - Solution parameters:')
print('a_result     = %1.2e \t a_solution     = %1.2e' % (result.x[0], a_sol))
print('r_result     = %1.2e \t r_solution     = %1.2e' % (result.x[1], r_sol) )
print('eps_result   = %1.2e \t eps_solution   = %1.2e' % (result.x[2], eps_sol) )
print('delta_result = %1.2e \t delta_solution = %1.2e' % (result.x[3], dep_col_sol) )
print('Hd_result    = %1.2e \t Hd_solution    = %1.2e' % (result.x[4], Hd_sol) )

I_res, Q_res = fs.solve_profiles(result.x[0], result.x[1], result.x[2], result.x[3], result.x[4])

plt.plot((I_sol)[-1, :, -1], 'ok', label=r'$I/B_{\nu}$ "observed"')
plt.plot((I_initial)[-1, :, -1], 'r', label=r'$I/B_{\nu}$ initial')
plt.plot((I_res)[-1, :, -1], 'b', label=r'$I/B_{\nu}$ inverted')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()
plt.plot((Q_initial)[-1, :, -1], 'r--', label='$Q/I$ initial')
plt.plot((Q_sol)[-1, :, -1], 'ok', label='$Q/I$ "observed"')
plt.plot((Q_res)[-1, :, -1], 'b--', label='$Q/I$ inverted')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()