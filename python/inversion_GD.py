#############################################################################
#    INVERSION CODE EXAMPLE USING FORWARD_SOLVER MODEL ATMOSPHERE           #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
import forward_solver_py as fs
import parameters as pm

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
    a, r, eps, dep_col, Hd = params[0], params[1], params[2], params[3], params[4]

    I, Q = fs.solve_profiles(a, r, eps, dep_col, Hd)

    I_obs = I[-1,:,mu].copy()
    Q_obs = Q[-1,:,mu].copy()

    chi2 = np.sum(w_I*(I_obs-I_obs_sol)**2/std**2 + w_Q*(Q_obs-Q_obs_sol)**2/std**2)/(2*I_obs.size)

    print(f'Total Chi^2 of this profiles is: {chi2}')
    return chi2, I_obs, Q_obs #, Jm00_new, Jm20_new


def surroundings(x_0, h):

    surroundings = np.zeros((x_0.shape[0], *x_0.shape))
    for i in range(x_0.shape[0]):
        delta = np.zeros_like(x_0)
        delta[i] = delta[i] + h
        surroundings[i] = x_0 + delta

    return surroundings


def compute_gradient(I_sol, Q_sol, I_0, Q_0, x_0, xs, std, w_I, w_Q, mu):

    chi2_pivot, _ , _  = chi2(x_0, I_sol, Q_sol, std, w_I, w_Q, mu)
    chi2s = np.ones((x_0.shape[0],2))
    chi2s[:,0] = np.ones(x_0.shape[0])*chi2_pivot
    
    for i in range(len(x_0)):
        chi2s[i,1], _ , _ = chi2(xs[i], I_sol, Q_sol, std, w_I, w_Q, mu)

    _ , beta = np.gradient(chi2s,1)
    beta = beta[:,0]

    return beta

####################    COMPUTE THE "OBSERVED" PROFILE    #####################
a_sol = 1e-3      #1e-3                 # dumping Voigt profile a=gam/(2^1/2*sig)
r_sol = 1e-2      #1e-2                 # XCI/XLI
eps_sol = 1e-1    #1e-1                 # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col_sol = 8   #8                    # Depolirarization colisions (delta)
Hd_sol = .8       #.8                  # Hanle depolarization factor [1/5, 1]
mu = 9 #int(fs.pm.qnd/2)

##############      INITIALICE THE PARAMETERS       #######################
seed = 666
np.random.seed(seed)
a_initial =  10**(-np.random.uniform(0,10))
r_initial =  10**(-np.random.uniform(0,12))
eps_initial = 10**(-np.random.uniform(0,4))
dep_col_initial =  np.random.uniform(0,1)
Hd_initial =  np.random.uniform(1/5, 1)

w_I     , w_Q   = 1e-1  , 1e3

h = 1e-8
max_itter = 250
std = 1e-5
# step_size = 1e-5
step_size = np.array([1e-4,1e-4,1e-4,1e0,1e0])


###################     COMPUTE THE SOLUTION PROFILES AND RADIATION FIELD       ###################
print("Solution parameters: ")
print(f" a = {a_sol}\n r = {r_sol}\n eps = {eps_sol}\n delta = {dep_col_sol}\n Hd = {Hd_sol}\n")
print("Computing the solution profiles:")

I_sol, Q_sol = fs.solve_profiles(a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol)
I_sol = I_sol[-1,:,mu].copy()
Q_sol = Q_sol[-1,:,mu].copy()

if(np.min(I_sol) < 0):
    print('Bad solution parameters, stopping.')
    exit()

### AD NOISE TO THE "OBSERVED" PROFILE
I_sol = add_noise(I_sol, std)
Q_sol = add_noise(Q_sol, std)

################     COMPUTE THE INITIAL GUESS PROFILES AND RADIATION FIELD      ###############
print("\nInitial parameters: ")
print(f" a = {a_initial}\n r = {r_initial}\n eps = {eps_initial}\n delta = {dep_col_initial}\n Hd = {Hd_initial}\n")
##########  MINIMIZE THE CHI2 FUNCTION WITH GIVEN RANGE CONSTRAINS #########

x_0 = np.array([a_initial,r_initial,eps_initial,dep_col_initial,Hd_initial])
x_l = np.array([1e-12,1e-12,1e-4,0,0.2])
x_u = np.array([1,1,1,10,1])

points = x_0.copy()

chi2_0, I_0, Q_0 = chi2(x_0, I_sol, Q_sol, std, w_I, w_Q, mu)

for itt in range(max_itter):
# calculation of the drerivatives of the forward model
    
    print(f'itteration {itt} with a step size of {step_size}')
    print("\nNew parameters: ")
    print(f" a = {x_0[0]}\n r = {x_0[1]}\n eps = {x_0[2]}\n delta = {x_0[3]}\n Hd = {x_0[4]}\n")

    points = np.vstack((points,x_0))
    xs = surroundings(x_0, h)
    beta = compute_gradient(I_sol, Q_sol, I_0, Q_0, x_0, xs, std, w_I, w_Q, mu)

    x_1 = x_0 - step_size*beta

    for i in range(len(x_0)):
        if x_1[i] > x_u[i]:
            x_1[i] = x_u[i]
        elif x_1[i] < x_l[i]:
            x_1[i] = x_l[i]

    chi2_1, I_1, Q_1 = chi2(x_1, I_sol, Q_sol, std, w_I, w_Q, mu)

    if chi2_1 < 1e3:
        break

    x_0 = x_1.copy()
    chi2_0 = chi2_1.copy()
    I_0 = I_1.copy()
    Q_0 = Q_1.copy()

##### PRINT AND PLOT THE SOLUTION AND COMPARE IT TO THE INITIAL AND OBSERVED PROFILES ####
print("Computing the initial and final profiles:")

I_initial, Q_initial = fs.solve_profiles(a_initial, r_initial, eps_initial, dep_col_initial, Hd_initial)
I_initial = I_initial[-1,:,mu].copy()
Q_initial = Q_initial[-1,:,mu].copy()

a_res, r_res, eps_res, dep_col_res, Hd_res = x_0[0], x_0[1], x_0[2], x_0[3], x_0[4]
I_res, Q_res = fs.solve_profiles(a_res, r_res, eps_res, dep_col_res, Hd_res)
I_res = I_res[-1,:,mu].copy()
Q_res = Q_res[-1,:,mu].copy()

print('\nFound Parameters - Solution parameters:')
print('a_result     = %1.2e \t a_solution     = %1.2e \t a_initial      = %1.2e' % (a_res, a_sol, a_initial))
print('r_result     = %1.2e \t r_solution     = %1.2e \t r_initial      = %1.2e' % (r_res, r_sol, r_initial) )
print('eps_result   = %1.2e \t eps_solution   = %1.2e \t eps_initial    = %1.2e' % (eps_res, eps_sol, eps_initial) )
print('delta_result = %1.2e \t delta_solution = %1.2e \t delta_initial  = %1.2e' % (dep_col_res, dep_col_sol, dep_col_initial) )
print('Hd_result    = %1.2e \t Hd_solution    = %1.2e \t Hd_initial     = %1.2e\n' % (Hd_res, Hd_sol, Hd_initial) )
print('w_I          = %1.2e \t w_Q            = %1.2e' % (w_I, w_Q) )
print('mu           = %1.2e ' % mu )
print('std          = %1.2e ' % std )

plt.plot(I_sol, 'ok', label=r'$I/B_{\nu}$ "observed"')
plt.plot(I_initial, 'r', label=r'$I/B_{\nu}$ initial parameters')
plt.plot(I_res, 'b', label=r'$I/B_{\nu}$ inverted parameters')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()
plt.plot(Q_initial, 'r--', label='$Q/I$ initial parameters')
plt.plot(Q_sol, 'ok', label='$Q/I$ "observed"')
plt.plot(Q_res, 'b--', label='$Q/I$ inverted parameters')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()

plt.plot(points[:,2],points[:,4],'o-.k', markersize=3, linewidth=1)
plt.plot(points[-1,2],points[-1,4],'ok', markersize=7)
plt.plot(eps_sol,Hd_sol,'or', markersize=10)
plt.plot(eps_initial,Hd_initial,'ob', markersize=10)
plt.xlim(x_l[2],x_u[2]);    plt.ylim(x_l[4],x_u[4])
plt.title('Movement in the parameter space of the inversion vs solution')
plt.xlabel('eps');          plt.ylabel('H_d')
plt.xscale('log')
plt.show()

plt.plot(points[:,2],points[:,3],'o-.k', markersize=3, linewidth=1)
plt.plot(points[-1,2],points[-1,3],'ok', markersize=7)
plt.plot(eps_sol,dep_col_sol,'or', markersize=10)
plt.plot(eps_initial,dep_col_initial,'ob', markersize=10)
plt.xlim(x_l[2],x_u[2]);    plt.ylim(x_l[3],x_u[3])
plt.title('Movement in the parameter space of the inversion vs solution')
plt.xlabel('eps');          plt.ylabel('dep_col')
plt.xscale('log')
plt.show()

plt.plot(points[:,3],points[:,4],'o-.k', markersize=3, linewidth=1)
plt.plot(points[-1,3],points[-1,4],'ok', markersize=7)
plt.plot(dep_col_sol,Hd_sol,'or', markersize=10)
plt.plot(dep_col_initial,Hd_initial,'ob', markersize=10)
plt.xlim(x_l[3],x_u[3]);    plt.ylim(x_l[4],x_u[4])
plt.title('Movement in the parameter space of the inversion vs solution')
plt.xlabel('dep_col');      plt.ylabel('H_d')
plt.show()

plt.plot(points[:,1],points[:,2],'o-.k', markersize=3, linewidth=1)
plt.plot(points[-1,1],points[-1,2],'ok', markersize=7)
plt.plot(r_sol,eps_sol,'or', markersize=10)
plt.plot(r_initial,eps_initial,'ob', markersize=10)
plt.xlim(x_l[1],x_u[1]);    plt.ylim(x_l[2],x_u[2])
plt.title('Movement in the parameter space of the inversion vs solution')
plt.xlabel('r');            plt.ylabel('eps')
plt.xscale('log');          plt.yscale('log')
plt.show()