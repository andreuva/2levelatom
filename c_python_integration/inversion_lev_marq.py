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
    a, r, eps, dep_col, Hd = params

    # print("\n New parameters: ")
    # print(f" a = {a}\n r = {r}\n eps = {eps}\n delta = {dep_col}\n Hd = {Hd}\n")
    # print("\n Computing the new profiles:")

    I,Q = fs.solve_profiles(a, r, eps, dep_col, Hd)
    I_obs = I[-1,:,mu].copy()
    Q_obs = Q[-1,:,mu].copy()
    chi2 = np.sum(w_I*(I_obs-I_obs_sol)**2/std**2 + w_Q*(Q_obs-Q_obs_sol)**2/std**2)/(2*I_obs.size)

    print(f'Chi^2 of this profiles is: {chi2} ' )
    return chi2, I_obs, Q_obs

def surroundings(x_0, h):

    surroundings = np.zeros((x_0.shape[0], *x_0.shape))
    for i in range(x_0.shape[0]):
        delta = np.zeros_like(x_0)
        delta[i] = delta[i] + h
        surroundings[i] = x_0 + delta

    return surroundings

def compute_alpha_beta(I_sol, Q_sol, I_0, Q_0, x_0, xs, std, w_I, w_Q, mu):

    Is = np.zeros( (x_0.size, *I_sol.shape) )
    Qs = np.zeros( (x_0.size, *Q_sol.shape) )
    dQ = Qs.copy()
    dI = Is.copy()

    beta = np.zeros_like(x_0)
    alpha = np.zeros( (x_0.size, x_0.size) )
    
    for i in range(len(x_0)):
        I, Q = fs.solve_profiles( xs[i,0], xs[i,1], xs[i,2], xs[i,3], xs[i,4] )
        Is[i] = I[-1,:,mu].copy()
        Qs[i] = Q[-1,:,mu].copy()

        dI[i] = (I_0 - Is[i])/h
        dQ[i] = (Q_0 - Qs[i])/h

    for i in range(len(x_0)):
        beta[i] = np.sum( w_I*(Is[i] - I_sol)*dI[i] + w_Q*(Qs[i]-Q_sol)*dQ[i])/(I_sol.size*std**2)
        for j in range(len(x_0)):
            alpha[i,j] = np.sum( w_I*dI[i]*dI[j] + w_Q*dQ[i]*dQ[j])/(I_sol.size)

    return alpha, beta

####################    COMPUTE THE "OBSERVED" PROFILE    #####################
a_sol = 1e-3      #1e-5,1e-2 ,1e-4                # dumping Voigt profile a=gam/(2^1/2*sig)
r_sol = 1e-4      #1,1e-4   ,1e-10                 # XCI/XLI
eps_sol = 1e-4                          # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col_sol = 1e-2             #0.1          # Depolirarization colisions (delta)
Hd_sol = 0.25                  #1          # Hanle depolarization factor [1/5, 1]

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
w_I = 1e-1
w_Q = 1e2
h = 1e-5
max_itter = 200
std = 1e-5
# initial guess of the lambda parameter
lambd = 1e-5
new_point = True
solutions = []

I_sol = add_noise(I_sol, std)
Q_sol = add_noise(Q_sol, std)

random.seed(12455)
a = random.uniform(1e-10,1)
r = random.uniform(1e-15,1)
eps = random.uniform(1e-4,1)
dep_col = random.uniform(0,10)
Hd = random.uniform(1/5, 1)
print("\nInitial parameters: ")
print(f" a = {a}\n r = {r}\n eps = {eps}\n delta = {dep_col}\n Hd = {Hd}\n")

##########  MINIMIZE THE CHI2 FUNCTION WITH GIVEN RANGE CONSTRAINS #########
x_0 = np.array([a,r,eps,dep_col,Hd])
x_l = np.array([1e-12,1e-12,1e-4,0,0.2])
x_u = np.array([1,1,1,10,1])

chi2_0, I_0, Q_0 = chi2(x_0, I_sol, Q_sol, std, w_I, w_Q, mu)

for itt in range(max_itter):
# calculation of the drerivatives of the forward model
    
    print(f'itteration {itt} with a lambda of {lambd}')
    if new_point:
        print("\nNew parameters: ")
        print(f" a = {x_0[0]}\n r = {x_0[1]}\n eps = {x_0[2]}\n delta = {x_0[3]}\n Hd = {x_0[4]}\n")
        xs = surroundings(x_0, h)
        alpha, beta = compute_alpha_beta(I_sol, Q_sol, I_0, Q_0, x_0, xs, std, w_I, w_Q, mu)


    alpha_p = alpha.copy()
    for i in range(len(alpha)):
        alpha_p[i,i] = alpha[i,i] * (1 + lambd)

    deltas = np.linalg.solve(alpha_p, beta)

    for i in range(len(x_0)):
        if x_0[i] + deltas[i] > x_u[i]:
            deltas[i] = x_u[i] - x_0[i]
        elif x_0[i] + deltas[i] < x_l[i]:
            deltas[i] = x_l[i] - x_0[i]
        else:
            pass

    chi2_1, I_1, Q_1 = chi2(x_0 + deltas, I_sol, Q_sol, std, w_I, w_Q, mu)

    if chi2_1 >= chi2_0:
        lambd = lambd*10
        new_point = False

        if lambd > 1e25:
            if chi2_1 < 1e3:
                break
            else:
                lambd = 1e-5
                
                break

                # solutions.append( [x_0, chi2_1] )
                # if len(solutions) > 3:
                #     break

                # print('Solution has been stuck for some iterations.')
                # print('restarting with new parameters')
                # a = random.uniform(1e-10,1)
                # r = random.uniform(1e-15,1)
                # eps = random.uniform(1e-4,1)
                # dep_col = random.uniform(0,10)
                # Hd = random.uniform(1/5, 1)
                # x_0 = np.array([a,r,eps,dep_col,Hd])
                # new_point = True

                # chi2_0, I_0, Q_0 = chi2(x_0, I_sol, Q_sol, std, w_I, w_Q, mu)
                
                # xs = surroundings(x_0, h)
    else:
        lambd = lambd/10
        x_0 = x_0 + deltas
        chi2_0 = chi2_1
        I_0 = I_1
        Q_0 = Q_1
        new_point = True

a_res, r_res, eps_res, dep_col_res, Hd_res = x_0
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
plt.savefig('../figures/lev_marq_I.png')
plt.close()

plt.plot(Q_initial, 'r--', label='$Q/I$ initial')
plt.plot(Q_sol, 'ok', label='$Q/I$ "observed"')
plt.plot(Q_res, 'b--', label='$Q/I$ inverted')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.savefig('../figures/lev_marq_Q.png')
plt.close()