#############################################################################
#    INVERSION CODE EXAMPLE USING FORWARD_SOLVER MODEL ATMOSPHERE           #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
import forward_solver_py  as fs
import random

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
    a, r, eps, dep_col, Hd = 10**(-params[0]), 10**(-params[1]), 10**(-params[2]), params[3], params[4]

    # print("\n New parameters: ")
    # print(f" a = {a}\n r = {r}\n eps = {eps}\n delta = {dep_col}\n Hd = {Hd}\n")
    # print("\n Computing the new profiles:")

    I,Q = fs.solve_profiles(a, r, eps, dep_col, Hd)
    I_obs = I[-1,:,mu].copy()
    Q_obs = Q[-1,:,mu].copy()
    chi2 = np.sum(w_I*(I_obs-I_obs_sol)**2/std**2 + w_Q*(Q_obs-Q_obs_sol)**2/std**2)/(2*I_obs.size)

    print(f'Chi^2 of this profiles is: {chi2} ' )
    return chi2, I_obs, Q_obs

####################    COMPUTE THE "OBSERVED" PROFILE    #####################
a_sol = 1e-4      #1e-5,1e-2 ,1e-4                # dumping Voigt profile a=gam/(2^1/2*sig)
r_sol = 1e-2      #1,1e-4   ,1e-10                 # XCI/XLI
eps_sol = 1e-3                          # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col_sol = 1             #0.1          # Depolirarization colisions (delta)
Hd_sol = .2                  #1          # Hanle depolarization factor [1/5, 1]

mu = 9 #int(fs.pm.qnd/2)

print("Solution parameters: ")
print(f" a = {a_sol}\n r = {r_sol}\n eps = {eps_sol}\n delta = {dep_col_sol}\n Hd = {Hd_sol}\n")
print("Computing the solution profiles:")
I_sol, Q_sol = fs.solve_profiles(a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol)
I_sol = I_sol[-1,:,mu].copy()
Q_sol = Q_sol[-1,:,mu].copy()

if(np.min(I_sol) < 0):
    print('Bad solution parameters, stopping.')
    exit()

##############  INITIALICE THE PARAMETERS AND ADD NOISE TO "OBSERVED" #########
w_I = 1e-8
w_Q = 1e-5
max_itter = 100
nsamples = 5000
std = 1e-5

I_sol = add_noise(I_sol, std)
Q_sol = add_noise(Q_sol, std)

x_l = np.array([0,0,0,0,0.2])
x_u = np.array([12,12,4,1,1])

random.seed(12455)
a = random.uniform(x_l[0], x_u[0])
r = random.uniform(x_l[1], x_u[1])
eps = random.uniform(x_l[2], x_u[2])
dep_col = random.uniform(x_l[3], x_u[3])
Hd = random.uniform(x_l[4], x_u[4])

x_0 = np.array([a,r,eps,dep_col,Hd])
#############################################################################

chain = x_0.copy()

chi2_0, I_0, Q_0 = chi2(x_0, I_sol, Q_sol, std, w_I, w_Q, mu)
likelihood_0 = np.exp(-1/2 * chi2_0)
posterior_0 = likelihood_0

while chain.shape[0] < nsamples:

    a = random.uniform(x_l[0], x_u[0])
    r = random.uniform(x_l[1], x_u[1])
    eps = random.uniform(x_l[2], x_u[2])
    dep_col = random.uniform(x_l[3], x_u[3])
    Hd = random.uniform(x_l[4], x_u[4])
    x_1 = np.array([a,r,eps,dep_col,Hd])

    chi2_1, _, _ = chi2(x_1, I_sol, Q_sol, std, w_I, w_Q, mu)
    
    likelihood_1 = np.exp(-1/2 * chi2_1)
    posterior_1 = likelihood_1
    
    print(posterior_1, posterior_0)
    rr = posterior_1/posterior_0

    prob = np.min([1,rr])

    print('probability of acceptance: ', prob)

    is_selected = random.choices([True,False], weights=(prob,1-prob), k=1)[0]

    if is_selected:
        chain = np.vstack((chain, x_1))
        print(f"\nAdded point number {chain.shape[0]} to the chain: ")
        print(f" a = {10**(-a)}\n r = {10**(-r)}\n eps = {10**(-eps)}\n delta = {dep_col}\n Hd = {Hd}\n")
        x_0 = x_1.copy()
        posterior_0 = posterior_1
    else:
        if np.all(chain[-1] == x_0):
            print("x_0 already in the chain, skipping\n")
        else: 
            chain = np.vstack((chain, x_0))
            print(f"\nAdded point number {chain.shape[0]} to the chain: ")
            print(f" a = {10**(-x_0[0])}\n r = {10**(-x_0[1])}\n eps = {10**(-x_0[2])}\n delta = {x_0[3]}\n Hd = {x_0[4]}\n")


a_res, r_res, eps_res, dep_col_res, Hd_res = 10**(-x_0[0]), 10**(-x_0[1]), 10**(-x_0[2]), x_0[3], x_0[4]
##### PRINT AND PLOT THE SOLUTION AND COMPARE IT TO THE INITIAL AND OBSERVED PROFILES ####
print("Computing the initial and final profiles:")
I_initial, Q_initial = fs.solve_profiles(a, r, eps, dep_col, Hd)
I_initial = I_initial[-1,:,mu].copy()
Q_initial = Q_initial[-1,:,mu].copy()
I_res, Q_res = fs.solve_profiles(a_res, r_res, eps_res, dep_col_res, Hd_res)
I_res = I_res[-1,:,mu].copy()
Q_res = Q_res[-1,:,mu].copy()

print('\nFound Parameters - Solution parameters:')
print('a_result     = %1.2e \t a_solution     = %1.2e' % (a_res, a_sol))
print('r_result     = %1.2e \t r_solution     = %1.2e' % (r_res, r_sol) )
print('eps_result   = %1.2e \t eps_solution   = %1.2e' % (eps_res, eps_sol) )
print('delta_result = %1.2e \t delta_solution = %1.2e' % (dep_col_res, dep_col_sol) )
print('Hd_result    = %1.2e \t Hd_solution    = %1.2e\n' % (Hd_res, Hd_sol) )
print('w_I          = %1.2e \t w_Q            = %1.2e' % (w_I, w_Q) )
print('mu           = %1.2e ' % mu )
print('std          = %1.2e ' % std )

plt.plot(I_sol, 'ok', label=r'$I/B_{\nu}$ "observed"')
plt.plot(I_initial, 'r', label=r'$I/B_{\nu}$ initial')
plt.plot(I_res, 'b', label=r'$I/B_{\nu}$ inverted')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()
plt.plot(Q_initial, 'r--', label='$Q/I$ initial')
plt.plot(Q_sol, 'ok', label='$Q/I$ "observed"')
plt.plot(Q_res, 'b--', label='$Q/I$ inverted')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()

import corner
# Set up the parameters of the problem.
ndim, nsamples = x_0.size , chain.shape[0]

# Plot it.
figure = corner.corner(chain, labels=[r"$a$", r"$r$", r"$\epsilon$", r"$\delta$", r"H_d"],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 12})

figure.show()