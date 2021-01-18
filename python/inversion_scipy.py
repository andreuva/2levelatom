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

##########################     SUBROUTINES     ##############################
def add_noise(array, sigma):
    """
    Add nromal noise to an N dimensional array with a ginven sigma
    """
    noise = np.random.normal(0,sigma,array.size)
    noise = noise.reshape(array.shape)
    array = array + noise
    return array


def chi2(params, I_obs_sol, Q_obs_sol, std, w_I, w_Q, mu):
    '''
    Compute the cost function of the inversion given the parameters, the noise
    and weigths of the diferent components (I,Q...)
    '''
    a, r, eps, dep_col, Hd = params

    # print("\n New parameters: ")
    # print(f" a = {a}\n r = {r}\n eps = {eps}\n delta = {dep_col}\n Hd = {Hd}\n")
    print("\n Computing the new profiles:")

    I,Q = fs.solve_profiles(a, r, eps, dep_col, Hd)
    I_obs = I[-1,:,mu].copy()
    Q_obs = Q[-1,:,mu].copy()
    chi2 = np.sum(w_I*(I_obs-I_obs_sol)**2/std**2 + w_Q*(Q_obs-Q_obs_sol)**2/std**2)/(2*I_obs.size)

    print(f'Chi^2 of this profiles is: {np.max(np.abs(chi2))} ' )
    return chi2

####################    COMPUTE THE "OBSERVED" PROFILE    #####################
a_sol = 1e-6      #1e-5,1e-2 ,1e-4                # dumping Voigt profile a=gam/(2^1/2*sig)
r_sol = 1e-2         #1,1e-4   ,1e-10                 # XCI/XLI
eps_sol = 1e-4                          # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col_sol = 0             #0.1          # Depolirarization colisions (delta)
Hd_sol = 1                  #1          # Hanle depolarization factor [1/5, 1]

std = 1e-6
w_I = 1e-2
w_Q = 1e2
mu = -6 #int(fs.pm.qnd/2)

print(" Solution parameters: ")
print(f" a = {a_sol}\n r = {r_sol}\n eps = {eps_sol}\n delta = {dep_col_sol}\n Hd = {Hd_sol}\n")
print("Computing the solution profiles:")
I_sol, Q_sol = fs.solve_profiles(a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol)

I_sol = add_noise(I_sol, std)
Q_sol = add_noise(Q_sol, std)

I_obs_sol = I_sol[-1,:,mu].copy()
Q_obs_sol = Q_sol[-1,:,mu].copy()

I_obs_sol = add_noise(I_obs_sol, std)
Q_obs_sol = add_noise(Q_obs_sol, std)

if(np.min(I_obs_sol) < 0):
    print('Bad solution parameters, stopping.')
    exit()

##############  INITIALICE THE PARAMETERS AND COMPUTE INITIAL PROFILES #########
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
I_obs_initial = I_initial[-1,:,mu].copy()
Q_obs_initial = Q_initial[-1,:,mu].copy()

##########  MINIMIZE THE CHI2 FUNCTION WITH GIVEN RANGE CONSTRAINS #########
x0 = np.array([a,r,eps,dep_col,Hd])
xl = [1e-12,1e-12,1e-5,0,0.2]
xu = [1,1,1,10,1]
bounds = Bounds(xl,xu)
result = opt.minimize(chi2,x0, bounds=bounds, args=(I_obs_sol, Q_obs_sol, std, w_I, w_Q, mu))


##### PRINT AND PLOT THE SOLUTION AND COMPARE IT TO THE INITIAL AND OBSERVED PROFILES ####
print('\nFound Parameters - Solution parameters:')
print('a_result     = %1.2e \t a_solution     = %1.2e' % (result.x[0], a_sol))
print('r_result     = %1.2e \t r_solution     = %1.2e' % (result.x[1], r_sol) )
print('eps_result   = %1.2e \t eps_solution   = %1.2e' % (result.x[2], eps_sol) )
print('delta_result = %1.2e \t delta_solution = %1.2e' % (result.x[3], dep_col_sol) )
print('Hd_result    = %1.2e \t Hd_solution    = %1.2e\n' % (result.x[4], Hd_sol) )
print('w_I          = %1.2e \t w_Q            = %1.2e' % (w_I, w_Q) )
print('mu           = %1.2e ' % mu )
print('std          = %1.2e ' % std )

I_res, Q_res = fs.solve_profiles(result.x[0], result.x[1], result.x[2], result.x[3], result.x[4])
I_obs_res = I_res[-1,:,mu].copy()
Q_obs_res = Q_res[-1,:,mu].copy()

plt.plot(I_obs_sol, 'ok', label=r'$I/B_{\nu}$ "observed"')
plt.plot(I_obs_initial, 'r', label=r'$I/B_{\nu}$ initial')
plt.plot(I_obs_res, 'b', label=r'$I/B_{\nu}$ inverted')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()
plt.plot(Q_obs_initial, 'r--', label='$Q/I$ initial')
plt.plot(Q_obs_sol, 'ok', label='$Q/I$ "observed"')
plt.plot(Q_obs_res, 'b--', label='$Q/I$ inverted')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()