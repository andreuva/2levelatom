#############################################################################
#    INVERSION CODE EXAMPLE USING FORWARD_SOLVER MODEL ATMOSPHERE           #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
from scipy.interpolate import PchipInterpolator as mon_spline_interp
import matplotlib.pyplot as plt
import forward_solver_jkq  as fs
import forward_solver_py as sfs
import forward_solver_py_J as sfs_j
import parameters as pm
from multiprocessing import Pool

zz = np.arange(pm.zl, pm.zu + pm.dz, pm.dz)          # compute the 1D grid
z_nodes = zz[fs.selected]

##########################     SUBROUTINES     ##############################
def add_noise(array, sigma):
    """
    Add nromal noise to an N dimensional array with a ginven sigma
    """
    noise = np.random.normal(0,sigma,array.size)
    noise = noise.reshape(array.shape)
    array = array + noise
    return array


def chi2(params, I_obs_sol, Q_obs_sol, std, w_I, w_Q, w_j00, w_j20, mu):
    '''
    Compute the cost function of the inversion given the parameters, the noise
    and weigths of the diferent components (I,Q...)
    '''
    a, r, eps, dep_col, Hd = params[0], params[1], params[2], params[3], params[4]
    Jm00, Jm20 = params[5:-fs.nodes_len], params[-fs.nodes_len:]

    I, Q, Jm00_new, Jm20_new = fs.solve_profiles(a, r, eps, dep_col, Hd, Jm00, Jm20)

    I_obs = I[-1,:,mu].copy()
    Q_obs = Q[-1,:,mu].copy()

    chi2_p = np.sum(w_I*(I_obs-I_obs_sol)**2/std**2 + w_Q*(Q_obs-Q_obs_sol)**2/std**2)/(2*I_obs.size)
    chi2_r = np.sum(w_j00*(Jm00-Jm00_new)**2 + w_j20*(Jm20 - Jm20_new)**2 )/(2*fs.nodes_len)
    chi2 = chi2_p + chi2_r

    print(f'Chi^2 profiles: {chi2_p}\t Chi^2 regularization: {chi2_r}')
    print(f'Total Chi^2 of this profiles is: {chi2}')

    return chi2, I_obs, Q_obs, Jm00_new, Jm20_new


def surroundings(x_0, h):

    surroundings = np.zeros((x_0.shape[0], *x_0.shape))
    for i in range(x_0.shape[0]):
        delta = np.zeros_like(x_0)
        delta[i] = delta[i] + h
        surroundings[i] = x_0 + delta

    return surroundings


def chi2_g(i):
    res, _, _, _, _ = chi2(xs[i], I_sol, Q_sol, std, w_I, w_Q, w_j00, w_j20, mu)
    return res


def compute_gradient(I_sol, Q_sol, I_0, Q_0, x_0, xs, std, w_I, w_Q, w_j00, w_j20, mu):

    chi2_pivot, _, _, _, _ = chi2(x_0, I_sol, Q_sol, std, w_I, w_Q, w_j00, w_j20, mu)
    chi2s = np.ones((x_0.shape[0],2))
    chi2s[:,0] = np.ones(x_0.shape[0])*chi2_pivot
    
    # for i in range(len(x_0)):
    #     chi2s[i,1], _, _ , _, _ = chi2(xs[i], I_sol, Q_sol, std, w_I, w_Q, w_j00, w_j20, mu)

    with Pool(processes=10) as pool:
        chi2s_pre = pool.map(chi2_g, range(len(x_0)))

    for i in range(len(chi2s_pre)):
        chi2s[i,1] = chi2s_pre[i]

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

w_I     , w_Q   = 1e0  , 1e3
w_j00   , w_j20 = 1e5   , 1e10

h = 1e-8
max_itter = 1000
std = 1e-5
# step_size = 1e-4
step_size = np.array([1e-3,1e-3,1e-3,5,5e-1])
step_size = np.append(step_size,np.ones(sum(fs.selected*2))*1e-2)

Jm00_initial = np.array([1. , 0.99995944, 0.99983577, 0.99962606, 0.99932733, 0.99893665, 0.99845105, 0.99786758, 0.99718328, 0.97773623, 0.92565889, 0.84813991, 0.75236794, 0.64553164, 0.53481965, 0.38480041, 0.26508522, 0.18169505, 0.12148064, 0.08577316, 0.06477453, 0.05275551, 0.04591633, 0.04215418, 0.04019498, 0.03922855, 0.0387448 , 0.03860897, 0.03850076, 0.03841888,0.038362  , 0.03832884, 0.03831807])
Jm20_initial = np.array([ 1.44731191e-16, -2.81879436e-08, -1.14020509e-07, -2.59400797e-07, -4.66231909e-07, -7.36416945e-07, -1.07185901e-06, -1.47446120e-06, -1.94612661e-06, -1.05213095e-05, -3.32016139e-05, -6.69396683e-05, -1.08688101e-04, -1.55399539e-04, -2.04026612e-04, -2.70746465e-04, -3.27195292e-04, -3.36229008e-04, -3.39271150e-04, -7.81891276e-05, 4.50362171e-04,  1.10069299e-03,  1.70089796e-03,  2.12759684e-03, 2.37858629e-03,  2.50977815e-03,  2.57686847e-03,  2.59601123e-03, 2.61129167e-03,  2.62288006e-03,  2.63094668e-03,  2.63566178e-03, 2.63719564e-03])
Jm00_initial = Jm00_initial[fs.selected]*0
Jm20_initial = Jm20_initial[fs.selected]*0

Jml, Jmu = Jm00_initial/Jm00_initial * -1e20 , Jm00_initial/Jm00_initial * 1e2 

###################     COMPUTE THE SOLUTION PROFILES AND RADIATION FIELD       ###################
print("Solution parameters: ")
print(f" a = {a_sol}\n r = {r_sol}\n eps = {eps_sol}\n delta = {dep_col_sol}\n Hd = {Hd_sol}\n")
print("Computing the solution profiles:")
I_sol, Q_sol, Jm00_sol, Jm20_sol = sfs_j.solve_profiles(a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol)
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
x_0 = np.array([a_initial,r_initial,eps_initial,dep_col_initial,Hd_initial, *Jm00_initial, *Jm20_initial])
x_l = np.array([1e-12,1e-12,1e-4,0,0.2, *Jml, *Jml])
x_u = np.array([1,1,1,10,1, *Jmu, *Jmu])

points = x_0.copy()

chi2_0, I_0, Q_0, Jm00_0, Jm20_0 = chi2(x_0, I_sol, Q_sol, std, w_I, w_Q, w_j00, w_j20, mu)

for itt in range(max_itter):
# calculation of the drerivatives of the forward model
    
    print(f'itteration {itt} with a step size of type array')
    print("\nNew parameters: ")
    print(f" a = {x_0[0]}\n r = {x_0[1]}\n eps = {x_0[2]}\n delta = {x_0[3]}\n Hd = {x_0[4]}\n")

    points = np.vstack((points,x_0))
    xs = surroundings(x_0, h)
    beta = compute_gradient(I_sol, Q_sol, I_0, Q_0, x_0, xs, std, w_I, w_Q, w_j00, w_j20, mu)

    if itt < 20:
        x_1 = x_0 - step_size*beta*10**(-20+itt)
    else:
        x_1 = x_0 - step_size*beta

    for i in range(len(x_0)):
        if x_1[i] > x_u[i]:
            x_1[i] = x_u[i]
        elif x_1[i] < x_l[i]:
            x_1[i] = x_l[i]

    chi2_1, I_1, Q_1, Jm00_1, Jm20_1 = chi2(x_1, I_sol, Q_sol, std, w_I, w_Q, w_j00, w_j20, mu)

    if chi2_1 < 1e3:
        break

    x_0 = x_1.copy()
    chi2_0 = chi2_1.copy()
    I_0 = I_1.copy()
    Q_0 = Q_1.copy()
    Jm00_0 = Jm00_1.copy()
    Jm20_0 = Jm20_1.copy()


##### PRINT AND PLOT THE SOLUTION AND COMPARE IT TO THE INITIAL AND OBSERVED PROFILES ####
print("Computing the initial and final profiles:")
I_initial, Q_initial, _ ,_ = fs.solve_profiles(a_initial, r_initial, eps_initial, dep_col_initial, Hd_initial, Jm00_initial, Jm20_initial)
I_initial, Q_initial = sfs.solve_profiles(a_initial, r_initial, eps_initial, dep_col_initial, Hd_initial)
I_initial = I_initial[-1,:,mu].copy()
Q_initial = Q_initial[-1,:,mu].copy()

a_res, r_res, eps_res, dep_col_res, Hd_res = x_0[0], x_0[1], x_0[2], x_0[3], x_0[4]
I_res, Q_res, _, _ = fs.solve_profiles(a_res, r_res, eps_res, dep_col_res, Hd_res, Jm00_0, Jm20_0)
I_res = I_res[-1,:,mu].copy()
Q_res = Q_res[-1,:,mu].copy()

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

interp_J00_jkq = mon_spline_interp(z_nodes, Jm00_1, extrapolate=True)
interp_J20_jkq = mon_spline_interp(z_nodes, Jm20_1, extrapolate=True)

Jm00_interp_jkq = interp_J00_jkq(zz)
Jm20_interp_jkq = interp_J20_jkq(zz)

plt.plot(zz,Jm00_interp_jkq,'-.b', label='Interpolated Js')
plt.plot(z_nodes,Jm00_1,'ob', label='Inverted nodes')
plt.plot(z_nodes,Jm00_initial,'ok', label='Initial guess')
plt.plot(zz,Jm00_sol,'-.r', label='solution Js')
plt.legend(); plt.title('$J_0^0$')
plt.show()
plt.plot(zz,Jm20_interp_jkq,'-.b', label='Interpolated Js')
plt.plot(z_nodes,Jm20_1,'ob', label='Inverted nodes')
plt.plot(z_nodes,Jm20_initial,'ok', label='Initial guess')
plt.plot(zz,Jm20_sol,'-.r', label='solution Js')
plt.legend(); plt.title('$J_0^2$')
plt.show()