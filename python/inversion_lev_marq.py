#############################################################################
#    INVERSION CODE EXAMPLE USING FORWARD_SOLVER MODEL ATMOSPHERE           #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
import forward_solver  as fs
import random
import scipy.optimize as opt
from scipy.optimize import Bounds

##########################     SUBROUTINES     ##############################
def add_noise(array, sigma):
    """
    Add nromal noise to an N dimensional array with a ginven sigma
    """
    noise = np.random.normal(0,sigma,array.size)
    noise = noise.reshape(array.shape)
    array = array + noise
    return array


def chi2(a, r, eps, dep_col, Hd, std, w_I, w_Q):
    '''
    Compute the cost function of the inversion given the parameters, the noise
    and weigths of the diferent components (I,Q...)
    '''

    # print("\n New parameters: ")
    # print(f" a = {a}\n r = {r}\n eps = {eps}\n delta = {dep_col}\n Hd = {Hd}\n")
    print("\n Computing the new profiles:")

    I,Q = fs.solve_profiles(a, r, eps, dep_col, Hd)
    chi2 = np.sum(w_I*(I-I_sol)**2/std**2 + w_Q*(Q-Q_sol)**2/std**2)/(2*I.size)

    print(f'Chi^2 of this profiles is: {np.max(np.abs(chi2))} ' )
    return chi2

####################    COMPUTE THE "OBSERVED" PROFILE    #####################
a_sol = 1e-4      #1e-5,1e-2 ,1e-4                # dumping Voigt profile a=gam/(2^1/2*sig)
r_sol = 1e-6      #1,1e-4   ,1e-10                 # XCI/XLI
eps_sol = 1e-1                          # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col_sol = 0             #0.1          # Depolirarization colisions (delta)
Hd_sol = 1                  #1          # Hanle depolarization factor [1/5, 1]

print(" Solution parameters: ")
print(f" a = {a_sol}\n r = {r_sol}\n eps = {eps_sol}\n delta = {dep_col_sol}\n Hd = {Hd_sol}\n")
print("Computing the solution profiles:")
I_sol, Q_sol = fs.solve_profiles(a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol)

if(np.min(I_sol) < 0):
    print('Bad solution parameters, stopping.')
    exit()

std = 1e-8
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
xl = np.array([1e-4,1e-12,1e-4,0,0.2])
xu = np.array([1,1,1,10,1])
chi2_0 = chi2(a, r, eps, dep_col, Hd, std, w_I, w_Q)


a_res, r_res, eps_res, dep_col_res, Hd_res = a, r, eps, dep_col, Hd
##### PRINT AND PLOT THE SOLUTION AND COMPARE IT TO THE INITIAL AND OBSERVED PROFILES ####
print('\nFound Parameters - Solution parameters:')
print('a_result     = %1.2e \t a_solution     = %1.2e' % (a_res, a_sol))
print('r_result     = %1.2e \t r_solution     = %1.2e' % (r_res, r_sol) )
print('eps_result   = %1.2e \t eps_solution   = %1.2e' % (eps_res, eps_sol) )
print('delta_result = %1.2e \t delta_solution = %1.2e' % (dep_col_res, dep_col_sol) )
print('Hd_result    = %1.2e \t Hd_solution    = %1.2e' % (Hd_res, Hd_sol) )

I_res, Q_res = fs.solve_profiles(a_res, r_res, eps_res, dep_col_res, Hd_res)

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