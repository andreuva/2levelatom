#############################################################################
#    INVERSION CODE EXAMPLE USING FORWARD_SOLVER MODEL ATMOSPHERE           #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
from scipy.interpolate import PchipInterpolator as mon_spline_interp
import matplotlib.pyplot as plt
import forward_solver_jkq  as fs
import forward_solver_py_J as sfs_j
import parameters as pm
from multiprocessing import Pool
import os
from datetime import datetime
from scipy.stats import norm
import shelve

##########################     SUBROUTINES     ##############################
def add_noise(array, sigma):
    """
    Add nromal noise to an N dimensional array with a ginven sigma
    """
    noise = np.random.normal(0,sigma,array.size)
    noise = noise.reshape(array.shape)
    array = array + noise
    return array


def chi2(params , I_obs_sol, Q_obs_sol, std, w_I, w_Q, w_j00, w_j20, epsI=1, epsQ=1, lambd=1):
    '''
    Compute the cost function of the inversion given the parameters,
    the observed profiles, the noise and weigths of the diferent components (I,Q)
    '''
    # store the params in local variables
    a, r, eps, dep_col, Hd = params[0], params[1], params[2], params[3], params[4]
    Jm00, Jm20 = params[5:-fs.nodes_len], params[-fs.nodes_len:]

    # Apply the forward solver and retrive the new profiles (in the correct shape) and JKQs
    I, Q, Jm00_new, Jm20_new = fs.solve_profiles(a, r, eps, dep_col, Hd, Jm00, Jm20)
    I_obs = I[-1,:,mu].copy()
    Q_obs = Q[-1,:,mu].copy()

    xx = np.linspace(norm.ppf(0.0001), norm.ppf(0.9999), pm.nw)
    wprof = norm.pdf(xx)
    xx = np.linspace(norm.ppf(0.0001), norm.ppf(0.9999), len(Jm00))
    wrad = norm.pdf(xx)

    # Compute the different parts of the cost function and add them together
    chi2_p = np.sum(w_I*wprof*(I_obs-I_obs_sol)**2/std**2 + w_Q*wprof*(Q_obs-Q_obs_sol)**2/std**2)/(2*I_obs.size)/(w_I + w_Q)

    chi2_r = np.sum(w_j00*wrad*(Jm00-Jm00_new)**2/epsI**2 + w_j20*wrad*(Jm20 - Jm20_new)**2/epsQ**2 )/(2*fs.nodes_len)/(w_j00 + w_j20)
    chi2 = (chi2_p + lambd*chi2_r)/(1 + lambd)

    # print(f'Chi^2 profiles: {chi2_p}\t Chi^2 regularization: {chi2_r}')
    # print(f'Total Chi^2 of this profiles is: {chi2}')

    # Return the diferent contributions of the cost function as well as the profiles and JKQs
    return chi2_p, chi2_r, chi2, I_obs, Q_obs, Jm00_new, Jm20_new


def surroundings(x_0, h):
    '''
    Given a point x_0 and a step h, compute an array of points cointaining the
    surroundings (forward) of this point in all the dimensions 
    '''
    surroundings = np.zeros((x_0.shape[0], *x_0.shape))
    for i in range(x_0.shape[0]):
        delta = np.zeros_like(x_0)
        delta[i] = delta[i] + h
        surroundings[i] = x_0 + delta

    return surroundings

def chi2_g(i):
    '''
    Just return the total cost function and parallelaize the derivates
    '''
    _, _, res, _, _, _, _ = chi2(xs[i], I_sol, Q_sol, std, w_I, w_Q, w_j00, w_j20)
    return res


def compute_gradient(I_sol, Q_sol, I_0, Q_0, x_0, xs, std, w_I, w_Q, w_j00, w_j20):
    '''
    Compute the gradient of the cost function given the parameters and his surroundings,
    as well as the needed imputs for the Chi2 function.
    '''

    # Compute the loss in the x_0 point and store it in an array to compute the gradient
    _, _, chi2_pivot, _, _, _, _ = chi2(x_0, I_sol, Q_sol, std, w_I, w_Q, w_j00, w_j20)
    chi2s = np.ones((x_0.shape[0],2))
    chi2s[:,0] = np.ones(x_0.shape[0])*chi2_pivot
    
    # Parallelize the loss calculation of all the points in the surroundings of x_0 in 10 processes
    with Pool(processes=10) as pool:
        chi2s_pre = pool.map(chi2_g, range(len(x_0)))

    # fill the array with the rest of the losses and computing the gradient
    for i in range(len(chi2s_pre)):
        chi2s[i,1] = chi2s_pre[i]

    _ , beta = np.gradient(chi2s,1)
    beta = beta[:,0]

    return beta

####################     SOLUTION PARAMETERS         #####################
a_sol = 1e-4      #1e-3                 # dumping Voigt profile a=gam/(2^1/2*sig)
r_sol = 1e-4      #1e-2                 # XCI/XLI
eps_sol = 1e-3    #1e-1                 # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col_sol = 1   #8                    # Depolirarization colisions (delta)
Hd_sol = .6       #.8                  # Hanle depolarization factor [1/5, 1]
mu = 9 #int(fs.pm.qnd/2)

##############      INITIALICE THE PARAMETERS       #######################
seed = 1196
itt = 0
np.random.seed(seed)
a_initial =  10**(-np.random.uniform(0,10))
r_initial =  10**(-np.random.uniform(0,12))
eps_initial = 10**(-np.random.uniform(0,4))
dep_col_initial =  np.random.uniform(0,1)
Hd_initial =  np.random.uniform(1/5, 1)

w_I     , w_Q   = 1     , 1e3
w_j00   , w_j20 = 1e6   , 1e8

# retrieve the 1D grid of the forward solver define with the params in params.py
zz = fs.zz
z_nodes = fs.z_nodes

h = 1e-8
max_itter = 2000
std = 1e-5
cc = 1
new_point = True

step_size = np.array([1e-2,1e-7,1e-7,1e-1,1e-1])
step_size = np.append(step_size,np.ones(sum(fs.selected*2))*1e-4)

# Create a directory to store the figures
directory = f'../figures/{datetime.now().strftime("%H%M%S%f")}_{max_itter}_' + \
            f'{w_I}_{w_Q:.1e}_{w_j00:.1e}_{w_j20:.1e}_{np.around(np.mean(step_size),2):.1e}/'

if not os.path.exists(directory):
    os.makedirs(directory)

# Initial J00 = 1 and initial J20 = 0
Jm00_initial = np.ones(fs.nodes_len)
Jm20_initial = np.zeros(fs.nodes_len)

###########     COMPUTE THE SOLUTION PROFILES AND RADIATION FIELD  (WITH STANDAR FORWARD SOLVER)     ###############
print("Solution parameters: ")
print(f" a = {a_sol}\n r = {r_sol}\n eps = {eps_sol}\n delta = {dep_col_sol}\n Hd = {Hd_sol}\n")
print("Computing the solution profiles ....")
I_sol, Q_sol, Jm00_sol, Jm20_sol = sfs_j.solve_profiles(a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol)
I_sol = I_sol[-1,:,mu].copy()
Q_sol = Q_sol[-1,:,mu].copy()

# Add noise to the solution profiles
I_sol = add_noise(I_sol, std)
Q_sol = add_noise(Q_sol, std)

# Compute the initial guess of the profiles with the initial parameters
print("\nInitial parameters: ")
print(f" a = {a_initial}\n r = {r_initial}\n eps = {eps_initial}\n delta = {dep_col_initial}\n Hd = {Hd_initial}\n")

# x_0 is the parameters vector x_l,x_u the lower and upper limits of the parameters
x_0 = np.array([a_initial,r_initial,eps_initial,dep_col_initial,Hd_initial, *Jm00_initial, *Jm20_initial])
x_0 = x_0
Jml = Jm00_initial/Jm00_initial * -1
Jmu = Jm00_initial/Jm00_initial * 1e2

x_l = np.array([1e-12,1e-12,1e-4,0,0.2, *Jml, *Jml])
x_u = np.array([1,1,1,10,1, *Jmu, *Jmu])

# initialice the loss points and compute the profiles/loss
chi2_evolution = np.array([[1e9,1e9,1e9],[1e9,1e9,1e9]])
chi2_p, chi2_r, chi2_0, I_0, Q_0, Jm00_0, Jm20_0 = chi2(x_0, I_sol, Q_sol, std, w_I, w_Q, w_j00, w_j20)

# initialice the variable with all the path of the params in the minimization
x_0_evolution = x_0.copy()
chi2_evolution = np.vstack((chi2_evolution,np.array([chi2_p,chi2_r,chi2_0])))
Jm00_evolution = Jm00_0.copy()
Jm20_evolution = Jm20_0.copy()
I_evolution = I_0.copy()
Q_evolution = Q_0.copy()

for itt in range(max_itter):
    
    print(f'itteration {itt}')

    # If we took a new point add it to the evolution and compute the surroundings and the gradient
    if new_point:
        x_0_evolution = np.vstack((x_0_evolution,x_0))
        xs = surroundings(x_0, h)
        beta = compute_gradient(I_sol, Q_sol, I_0, Q_0, x_0, xs, std, w_I, w_Q, w_j00, w_j20)

    # In the first steps go slow and then regular
    x_1 = x_0 - step_size*beta*cc

    # Check if the steps exceed the boundaris of the param space and if so, adjust the params
    for i in range(len(x_0)):
        if x_1[i] > x_u[i]:
            x_1[i] = x_u[i]
        elif x_1[i] < x_l[i]:
            x_1[i] = x_l[i]

    # Compute the profiles of the new point as well as the loss of that point
    chi2_p, chi2_r, chi2_1, I_1, Q_1, Jm00_1, Jm20_1 = chi2(x_1, I_sol, Q_sol, std, w_I, w_Q, w_j00, w_j20)
    print(f'Chi^2 profiles: {chi2_p}\t Chi^2 regularization: {chi2_r}')
    print(f'Total Chi^2 of this profiles is: {chi2_1}')

    # If we have very good loss stop
    if chi2_1 < 10:
        break

    # if the point improves the inversion take it and update the params and step size cc
    if chi2_1 < chi2_0 or itt==0:
        x_0 = x_1.copy()
        cc = cc*1.25
        new_point = True
        print("\nNew parameters: ")
        print(f" a = {x_0[0]}\n r = {x_0[1]}\n eps = {x_0[2]}\n delta = {x_0[3]}\n Hd = {x_0[4]}\n")
        
        chi2_0 = chi2_1.copy()
        chi2_evolution = np.vstack((chi2_evolution,np.array([chi2_p,chi2_r,chi2_0])))

        I_0 = I_1.copy()
        Q_0 = Q_1.copy()
        I_evolution = np.vstack((I_evolution, I_0))
        Q_evolution = np.vstack((Q_evolution, Q_0))

        Jm00_0 = Jm00_1.copy()
        Jm20_0 = Jm20_1.copy()
        Jm00_evolution = np.vstack((Jm00_evolution,Jm00_0))
        Jm20_evolution = np.vstack((Jm20_evolution,Jm20_0))
    # if the point is worse change the step size to one smaler in order to continue along the gradient
    else:
        cc = cc/2
        new_point = False


# Computing the interpolated final JKQs

interp_J00_jkq = mon_spline_interp(z_nodes, Jm00_evolution[-1], extrapolate=True)
interp_J20_jkq = mon_spline_interp(z_nodes, Jm20_evolution[-1], extrapolate=True)

Jm00_interp_jkq = interp_J00_jkq(zz)
Jm20_interp_jkq = interp_J20_jkq(zz)

# computing and the final solution and parameters
print("Computing the final profiles:")
a_res, r_res, eps_res, dep_col_res, Hd_res = x_0[0], x_0[1], x_0[2], x_0[3], x_0[4]
I_res = I_evolution[-1]
Q_res = Q_evolution[-1]

I_params, Q_params, Jm00_params, Jm20_params = sfs_j.solve_profiles(a_res, r_res, eps_res, dep_col_res, Hd_res)
I_params = I_params[-1,:,mu].copy()
Q_params = Q_params[-1,:,mu].copy()

I_initial = I_evolution[0]
Q_initial = Q_evolution[0]

####     SAVING ALL THE WORKSPACE    ####
filename = directory + 'workspace_variables.out'
my_shelf = shelve.open(filename,'n') # 'n' for new

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except:
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()


#############################   PRINTS AND PLOTS    #######################################
print('\nFound Parameters - Solution parameters:')
print('a_result     = %1.2e \t a_solution     = %1.2e \t a_initial      = %1.2e' % (a_res, a_sol, a_initial))
print('r_result     = %1.2e \t r_solution     = %1.2e \t r_initial      = %1.2e' % (r_res, r_sol, r_initial) )
print('eps_result   = %1.2e \t eps_solution   = %1.2e \t eps_initial    = %1.2e' % (eps_res, eps_sol, eps_initial) )
print('delta_result = %1.2e \t delta_solution = %1.2e \t delta_initial  = %1.2e' % (dep_col_res, dep_col_sol, dep_col_initial) )
print('Hd_result    = %1.2e \t Hd_solution    = %1.2e \t Hd_initial     = %1.2e\n' % (Hd_res, Hd_sol, Hd_initial) )
print('w_I          = %1.2e \t w_Q            = %1.2e' % (w_I, w_Q) )
print('w_J00        = %1.2e \t w_J20          = %1.2e' % (w_j00, w_j20) )
print('mu           = %1.2e ' % mu )
print('std          = %1.2e ' % std )

# Plot of the I and Q profiles. Observed, initial guess, inverted and regular FS with inverted params
# All the figures are save in the figures directory.
plt.plot(I_sol, 'ok', label=r'$I/B_{\nu}$ "observed"')
plt.plot(I_initial, 'r', label=r'$I/B_{\nu}$ initial parameters')
plt.plot(I_res, 'b', label=r'$I/B_{\nu}$ inverted parameters')
plt.plot(I_params, 'g', label=r'$I/B_{\nu}$ solution parameters')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.savefig(directory + 'I.png')
plt.close()
plt.plot(Q_initial, 'r', label='$Q$ initial parameters')
plt.plot(Q_sol, 'ok', label='$Q$ "observed"')
plt.plot(Q_res, 'b', label='$Q$ inverted parameters')
plt.plot(Q_params, 'g', label='$Q$ solution parameters')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.savefig(directory + 'Q.png')
plt.close()

# Plot the steps taken in different combinations of the parameters
# --------------------------------------------------------------------------------------
plt.plot(x_0_evolution[:,2],x_0_evolution[:,4],'o-.k', markersize=3, linewidth=1)
plt.plot(x_0_evolution[-1,2],x_0_evolution[-1,4],'ok', markersize=7)
plt.plot(eps_sol,Hd_sol,'or', markersize=10)
plt.plot(eps_initial,Hd_initial,'ob', markersize=10)
plt.xlim(x_l[2],x_u[2]);    plt.ylim(x_l[4],x_u[4])
plt.title('Movement in the parameter space of the inversion vs solution')
plt.xlabel('eps');          plt.ylabel('H_d')
plt.xscale('log')
plt.savefig(directory + 'eps_Hd.png')
plt.close()

plt.plot(x_0_evolution[:,2],x_0_evolution[:,3],'o-.k', markersize=3, linewidth=1)
plt.plot(x_0_evolution[-1,2],x_0_evolution[-1,3],'ok', markersize=7)
plt.plot(eps_sol,dep_col_sol,'or', markersize=10)
plt.plot(eps_initial,dep_col_initial,'ob', markersize=10)
plt.xlim(x_l[2],x_u[2]);    plt.ylim(x_l[3],x_u[3])
plt.title('Movement in the parameter space of the inversion vs solution')
plt.xlabel('eps');          plt.ylabel('dep_col')
plt.xscale('log')
plt.savefig(directory + 'eps_delta.png')
plt.close()

plt.plot(x_0_evolution[:,3],x_0_evolution[:,4],'o-.k', markersize=3, linewidth=1)
plt.plot(x_0_evolution[-1,3],x_0_evolution[-1,4],'ok', markersize=7)
plt.plot(dep_col_sol,Hd_sol,'or', markersize=10)
plt.plot(dep_col_initial,Hd_initial,'ob', markersize=10)
plt.xlim(x_l[3],x_u[3]);    plt.ylim(x_l[4],x_u[4])
plt.title('Movement in the parameter space of the inversion vs solution')
plt.xlabel('dep_col');      plt.ylabel('H_d')
plt.savefig(directory + 'delta_Hd.png')
plt.close()

plt.plot(x_0_evolution[:,1],x_0_evolution[:,2],'o-.k', markersize=3, linewidth=1)
plt.plot(x_0_evolution[-1,1],x_0_evolution[-1,2],'ok', markersize=7)
plt.plot(r_sol,eps_sol,'or', markersize=10)
plt.plot(r_initial,eps_initial,'ob', markersize=10)
plt.xlim(x_l[1],x_u[1]);    plt.ylim(x_l[2],x_u[2])
plt.title('Movement in the parameter space of the inversion vs solution')
plt.xlabel('r');            plt.ylabel('eps')
plt.xscale('log');          plt.yscale('log')
plt.savefig(directory + 'r_eps.png')
plt.close()
# --------------------------------------------------------------------------------------

# Plot the intitial, inverted, solution and computed with final parameters JKQs
plt.plot(zz,Jm00_interp_jkq,'-.b', label='Interpolated Js')
plt.plot(z_nodes,Jm00_evolution[-1],'ob', label='Inverted nodes')
plt.plot(z_nodes,Jm00_evolution[0],'ok', label='Initial guess')
plt.plot(zz,Jm00_sol,'-.r', label='solution Js')
plt.plot(zz,Jm00_params,'g', label='final params Js')
plt.legend(); plt.title('$J_0^0$')
plt.savefig(directory + 'Jm00.png')
plt.close()
plt.plot(zz,Jm20_interp_jkq,'-.b', label='Interpolated Js')
plt.plot(z_nodes,Jm20_evolution[-1],'ob', label='Inverted nodes')
plt.plot(z_nodes,Jm20_evolution[0],'ok', label='Initial guess')
plt.plot(zz,Jm20_sol,'-.r', label='solution Js')
plt.plot(zz,Jm20_params,'g', label='final params Js')
plt.legend(); plt.title('$J_0^2$')
plt.savefig(directory + 'Jm20.png')
plt.close()

# Print the evolution of the loss function with each itteration
plt.plot(chi2_evolution[4:,2], 'k', label=r'total $\chi^2$')
plt.plot(chi2_evolution[4:,0], 'b', label=r'$\chi^2$ of the profiles')
plt.plot(chi2_evolution[4:,1], 'r', label=r'$\chi^2$ of the regularization')
plt.legend(); plt.title(r'$\chi^2$ contributions')
plt.yscale('log')
plt.savefig(directory + 'chi2.png')
plt.close()


# Print the evolution of I, Q and JQKs in different itterations
jump = int(max_itter/4)
plt.plot(I_sol, 'ok', label=r'$I/B_{\nu}$ "observed"')
plt.plot(I_initial, 'r', label=r'$I/B_{\nu}$ initial parameters')
plt.plot(I_params, 'g', label=r'$I/B_{\nu}$ solution parameters')
plt.plot(I_evolution[-1],label=f'inverted parameters')
for i in range(0,max_itter,jump):
    plt.plot(I_evolution[i],label=f'itt {i}')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$'); 
plt.savefig(directory + 'I_evolution.png')
plt.close()

plt.plot(Q_sol, 'ok', label=r'$Q$ "observed"')
plt.plot(Q_initial, 'r', label=r'$Q$ initial parameters')
plt.plot(Q_params, 'g', label='$Q$ solution parameters')
plt.plot(Q_evolution[-1],label=f'inverted parameters')
for i in range(0,max_itter,jump):
    plt.plot(Q_evolution[i],label=f'itt {i}')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$'); 
plt.savefig(directory + 'Q_evolution.png')
plt.close()


# JKQs
interp_J00_jkq = mon_spline_interp(z_nodes, Jm00_evolution[-1], extrapolate=True)
Jm00_interp_jkq = interp_J00_jkq(zz)

plt.plot(zz,Jm00_interp_jkq,'-.', label=f'inverted parameters')
plt.plot(zz,Jm00_sol,'r', label='solution Js')
plt.plot(zz,Jm00_params,'g', label='final params Js')
for i in range(0,max_itter,jump):
    interp_J00_jkq = mon_spline_interp(z_nodes, Jm00_evolution[i], extrapolate=True)
    Jm00_interp_jkq = interp_J00_jkq(zz)
    plt.plot(zz,Jm00_interp_jkq,'-.', label=f'itt {i}')
plt.legend(); plt.title('$J_0^0$');
plt.savefig(directory + 'Jm00_evolution.png')
plt.close()

interp_J20_jkq = mon_spline_interp(z_nodes, Jm20_evolution[-1], extrapolate=True)
Jm20_interp_jkq = interp_J20_jkq(zz)

plt.plot(zz,Jm20_interp_jkq,'-.', label=f'inverted parameters')
plt.plot(zz,Jm20_sol,'r', label='solution Js')
plt.plot(zz,Jm20_params,'g', label='final params Js')
for i in range(0,max_itter,jump):
    interp_J20_jkq = mon_spline_interp(z_nodes, Jm20_evolution[i], extrapolate=True)
    Jm20_interp_jkq = interp_J20_jkq(zz)
    plt.plot(zz,Jm20_interp_jkq,'-.', label=f'itt {i}')
plt.legend(); plt.title('$J_0^2$');
plt.savefig(directory + 'Jm20_evolution.png')
plt.close()
