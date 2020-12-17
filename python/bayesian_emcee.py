#############################################################################
#    INVERSION CODE EXAMPLE USING FORWARD_SOLVER MODEL ATMOSPHERE           #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
import forward_solver_py  as fs
import random
import emcee
import corner

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

def log_prior(params, x_l, x_u):
    if x_l[0] < params[0] < x_u[0] and x_l[1] < params[1] < x_u[1] and x_l[2] < params[2] < x_u[2]\
        and x_l[3] < params[3] < x_u[3] and x_l[4] < params[4] < x_u[4]:
        return 0.0
    return -np.inf

def log_likelihood(params, I_sol, Q_sol, std, w_I, w_Q, mu):

    chi2_val, _, _ = chi2(params , I_sol, Q_sol, std, w_I, w_Q, mu)
    
    return -1/2 * chi2_val

def log_probability(params, x_l, x_u, I_sol, Q_sol, std, w_I, w_Q, mu):
    lp = log_prior(params, x_l, x_u)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, I_sol, Q_sol, std, w_I, w_Q, mu)

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
std = 1e-5

I_sol = add_noise(I_sol, std)
Q_sol = add_noise(Q_sol, std)

x_l = np.array([0,0,0,0,0.2])
x_u = np.array([12,12,4,1,1])
#############################################################################
ndim = 5
nsamples = 5000
nwalkers = 10

random.seed(1234)
x_0 = None
for i in range(nwalkers):
    a = random.uniform(x_l[0], x_u[0])
    r = random.uniform(x_l[1], x_u[1])
    eps = random.uniform(x_l[2], x_u[2])
    dep_col = random.uniform(x_l[3], x_u[3])
    Hd = random.uniform(x_l[4], x_u[4])

    init = np.array([a,r,eps,dep_col,Hd])
    if x_0 is None:
        x_0 = init.copy()
    else:
        x_0 = np.vstack((x_0, init))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, moves = emcee.moves.StretchMove(), args=(x_l, x_u, I_sol, Q_sol, std, w_I, w_Q, mu))
sampler.run_mcmc(x_0, nsamples, progress=True, skip_initial_state_check= True)

chain = sampler.get_chain(discard=250, thin=15, flat=True)

# Plot it.
figure = corner.corner(chain, labels=[r"$a$", r"$r$", r"$\epsilon$", r"$\delta$", r"H_d"],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 12})

figure.show()