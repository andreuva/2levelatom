#############################################################################
#    INVERSION CODE EXAMPLE USING FORWARD_SOLVER MODEL ATMOSPHERE           #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
import forward_solver_jkq  as fs
import forward_solver_py as sfs
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


def chi2(params, I_obs_sol, Q_obs_sol, std, w_I, w_Q, w_j00, w_j20, mu):
    '''
    Compute the cost function of the inversion given the parameters, the noise
    and weigths of the diferent components (I,Q...)
    '''
    a, r, eps, dep_col, Hd = params[0], params[1], params[2], params[3], params[4]
    Jm00, Jm20 = params[5:-fs.nodes_len], params[-fs.nodes_len:]

    # print("\n New parameters: ")
    # print(f" a = {a}\n r = {r}\n eps = {eps}\n delta = {dep_col}\n Hd = {Hd}\n")
    # print("\n Computing the new profiles:")

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


def compute_alpha_beta(I_sol, Q_sol, I_0, Q_0, x_0, xs, std, w_I, w_Q, w_j00, w_j20, mu):

    Is = np.zeros( (x_0.size, *I_sol.shape) )
    Qs = np.zeros( (x_0.size, *Q_sol.shape) )
    dQ = Qs.copy()
    dI = Is.copy()
    Jm00_p = x_0[5:-fs.nodes_len]
    Jm20_p = x_0[-fs.nodes_len:]
    Jm00s = np.zeros( (x_0.size, fs.nodes_len) )
    Jm20s = np.zeros( (x_0.size, fs.nodes_len) )
    dJm00 = Jm00s.copy()
    dJm20 = Jm20s.copy()

    beta = np.zeros_like(x_0)
    alpha = np.zeros( (x_0.size, x_0.size) )
    
    
    for i in range(len(x_0)):
        I, Q, Jm00s[i], Jm20s[i] = fs.solve_profiles( xs[i,0], xs[i,1], xs[i,2], xs[i,3], xs[i,4], xs[i,5:-fs.nodes_len], xs[i,-fs.nodes_len:] )
        Is[i] = I[-1,:,mu].copy()
        Qs[i] = Q[-1,:,mu].copy()

        dJm00[i] = (Jm00_p - Jm00s[i])/h
        dJm20[i] = (Jm20_p - Jm20s[i])/h
        dI[i] = (I_0 - Is[i])/h
        dQ[i] = (Q_0 - Qs[i])/h

    for i in range(len(x_0)):
        beta[i] = np.sum( w_I*(Is[i] - I_sol)*dI[i] + w_Q*(Qs[i]-Q_sol)*dQ[i])/(I_sol.size*std**2)\
                + np.sum(w_j00*(Jm00s[i] - Jm00_p)*dJm00[i] + w_j20*(Jm20s[i] - Jm20_p)*dJm20[i] )/fs.nodes_len
        for j in range(len(x_0)):
            alpha[i,j] = np.sum( w_I*dI[i]*dI[j] + w_Q*dQ[i]*dQ[j])/(I_sol.size)\
                       + np.sum( w_j00*dJm00[i]*dJm00[j] + w_j20*dJm20[i]*dJm20[j] )/fs.nodes_len

    return alpha, beta

####################    COMPUTE THE "OBSERVED" PROFILE    #####################
a_sol = 1e-4      #1e-5,1e-2 ,1e-4                # dumping Voigt profile a=gam/(2^1/2*sig)
r_sol = 1e-4      #1,1e-4   ,1e-10                 # XCI/XLI
eps_sol = 1e-3                          # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col_sol = .7             #0.1          # Depolirarization colisions (delta)
Hd_sol = .8                  #1          # Hanle depolarization factor [1/5, 1]
mu = 9 #int(fs.pm.qnd/2)

print("Solution parameters: ")
print(f" a = {a_sol}\n r = {r_sol}\n eps = {eps_sol}\n delta = {dep_col_sol}\n Hd = {Hd_sol}\n")
print("Computing the solution profiles:")
I_sol, Q_sol = sfs.solve_profiles(a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol)
I_sol = I_sol[-1,:,mu].copy()
Q_sol = Q_sol[-1,:,mu].copy()

if(np.min(I_sol) < 0):
    print('Bad solution parameters, stopping.')
    exit()

##############  INITIALICE THE PARAMETERS AND ADD NOISE TO "OBSERVED" #########
w_I, w_Q = 1e-1, 1e1
w_j00, w_j20 = 1e2, 1e6
h = 1e-5
max_itter = 1000
std = 1e-5
# initial guess of the lambda parameter
lambd = 1e0
new_point = True
solutions = []

I_sol = add_noise(I_sol, std)
Q_sol = add_noise(Q_sol, std)

np.random.seed(11196)
a =  a_sol                           #10**(-np.random.uniform(0,10))
r =  r_sol                           #10**(-np.random.uniform(0,12))
eps = eps_sol                        #10**(-np.random.uniform(0,4))
dep_col =  dep_col_sol               #np.random.uniform(0,1)
Hd =  Hd_sol                         #np.random.uniform(1/5, 1)

# Jm00 = np.arange(fs.pm.zl, fs.pm.zu + fs.pm.dz, fs.pm.dz)[fs.selected]
# Jm00 = (10 - Jm00)*1e-5
# Jm20 = np.ones_like(Jm00)*1e-5
# Jm00 = 10**(-np.random.uniform(0,2, fs.nodes_len))
# Jm20 = 10**(-np.random.uniform(3,8, fs.nodes_len))

Jm00 = np.array([1. , 0.99995944, 0.99983577, 0.99962606, 0.99932733, 0.99893665, 0.99845105, 0.99786758, 0.99718328, 0.97773623, 0.92565889, 0.84813991, 0.75236794, 0.64553164, 0.53481965, 0.38480041, 0.26508522, 0.18169505, 0.12148064, 0.08577316, 0.06477453, 0.05275551, 0.04591633, 0.04215418, 0.04019498, 0.03922855, 0.0387448 , 0.03860897, 0.03850076, 0.03841888,0.038362  , 0.03832884, 0.03831807])
Jm20 = np.array([ 1.44731191e-16, -2.81879436e-08, -1.14020509e-07, -2.59400797e-07, -4.66231909e-07, -7.36416945e-07, -1.07185901e-06, -1.47446120e-06, -1.94612661e-06, -1.05213095e-05, -3.32016139e-05, -6.69396683e-05, -1.08688101e-04, -1.55399539e-04, -2.04026612e-04, -2.70746465e-04, -3.27195292e-04, -3.36229008e-04, -3.39271150e-04, -7.81891276e-05, 4.50362171e-04,  1.10069299e-03,  1.70089796e-03,  2.12759684e-03, 2.37858629e-03,  2.50977815e-03,  2.57686847e-03,  2.59601123e-03, 2.61129167e-03,  2.62288006e-03,  2.63094668e-03,  2.63566178e-03, 2.63719564e-03])
Jm00 = Jm00[fs.selected]
Jm20 = Jm20[fs.selected]
Jm00_initial = Jm00.copy()
Jm20_initial = Jm20.copy()

Jml, Jmu = Jm00/Jm00 * -1e20 , Jm00/Jm00 * 1e2 

print("\nInitial parameters: ")
print(f" a = {a}\n r = {r}\n eps = {eps}\n delta = {dep_col}\n Hd = {Hd}\n")

##########  MINIMIZE THE CHI2 FUNCTION WITH GIVEN RANGE CONSTRAINS #########
x_0 = np.array([a,r,eps,dep_col,Hd, *Jm00, *Jm20])
x_l = np.array([1e-12,1e-12,1e-4,0,0.2, *Jml, *Jml])
x_u = np.array([1,1,1,10,1, *Jmu, *Jmu])
points = x_0.copy()

chi2_0, I_0, Q_0, Jm00_0, Jm20_0 = chi2(x_0, I_sol, Q_sol, std, w_I, w_Q, w_j00, w_j20, mu)

for itt in range(max_itter):
# calculation of the drerivatives of the forward model
    
    print(f'itteration {itt} with a lambda of {lambd}')
    if new_point:
        points = np.vstack((points,x_0))
        print("\nNew parameters: ")
        print(f" a = {x_0[0]}\n r = {x_0[1]}\n eps = {x_0[2]}\n delta = {x_0[3]}\n Hd = {x_0[4]}\n")
        xs = surroundings(x_0, h)
        alpha, beta = compute_alpha_beta(I_sol, Q_sol, I_0, Q_0, x_0, xs, std, w_I, w_Q, w_j00, w_j20, mu)


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

    chi2_1, I_1, Q_1, Jm00_1, Jm20_1 = chi2(x_0 + deltas, I_sol, Q_sol, std, w_I, w_Q, w_j00, w_j20, mu)

    if chi2_1 >= chi2_0:
        lambd = lambd*pm.dump_lev_marq
        new_point = False

        if lambd > 1e45:
            if chi2_1 < 1e3:
                break
            else:
                lambd = 1e-5
                
                break
    else:
        lambd = lambd/pm.dump_lev_marq
        x_0 = x_0 + deltas
        chi2_0 = chi2_1
        I_0 = I_1
        Q_0 = Q_1
        Jm00_0 = Jm00_1
        Jm20_0 = Jm20_1
        new_point = True

    if chi2_0 < 1e1:
        break

a_res, r_res, eps_res, dep_col_res, Hd_res = x_0[0], x_0[1], x_0[2], x_0[3], x_0[4]
##### PRINT AND PLOT THE SOLUTION AND COMPARE IT TO THE INITIAL AND OBSERVED PROFILES ####
print("Computing the initial and final profiles:")
I_initial, Q_initial, _ ,_ = fs.solve_profiles(a, r, eps, dep_col, Hd, Jm00_initial, Jm20_initial)
I_initial = I_initial[-1,:,mu].copy()
Q_initial = Q_initial[-1,:,mu].copy()
I_res, Q_res, _, _ = fs.solve_profiles(a_res, r_res, eps_res, dep_col_res, Hd_res, Jm00, Jm20)
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

plt.plot(points[:,2],points[:,4],'o-.k', markersize=3, linewidth=1)
plt.plot(points[-1,2],points[-1,4],'ok', markersize=7)
plt.plot(eps_sol,Hd_sol,'or', markersize=10)
plt.plot(eps,Hd,'ob', markersize=10)
plt.xlim(x_l[2],x_u[2]);    plt.ylim(x_l[4],x_u[4])
plt.xlabel('eps');          plt.ylabel('H_d')
plt.xscale('log')
plt.show()

plt.plot(points[:,2],points[:,3],'o-.k', markersize=3, linewidth=1)
plt.plot(points[-1,2],points[-1,3],'ok', markersize=7)
plt.plot(eps_sol,dep_col_sol,'or', markersize=10)
plt.plot(eps,dep_col,'ob', markersize=10)
plt.xlim(x_l[2],x_u[2]);    plt.ylim(x_l[3],x_u[3])
plt.xlabel('eps');          plt.ylabel('dep_col')
plt.xscale('log')
plt.show()

plt.plot(points[:,3],points[:,4],'o-.k', markersize=3, linewidth=1)
plt.plot(points[-1,3],points[-1,4],'ok', markersize=7)
plt.plot(dep_col_sol,Hd_sol,'or', markersize=10)
plt.plot(dep_col,Hd,'ob', markersize=10)
plt.xlim(x_l[3],x_u[3]);    plt.ylim(x_l[4],x_u[4])
plt.xlabel('dep_col');      plt.ylabel('H_d')
plt.show()

plt.plot(points[:,1],points[:,2],'o-.k', markersize=3, linewidth=1)
plt.plot(points[-1,1],points[-1,2],'ok', markersize=7)
plt.plot(r_sol,eps_sol,'or', markersize=10)
plt.plot(r,eps,'ob', markersize=10)
plt.xlim(x_l[1],x_u[1]);    plt.ylim(x_l[2],x_u[2])
plt.xlabel('r');            plt.ylabel('eps')
plt.xscale('log');          plt.yscale('log')
plt.show()

from scipy.interpolate import PchipInterpolator as mon_spline_interp

zz = np.arange(pm.zl, pm.zu + pm.dz, pm.dz)          # compute the 1D grid
z_nodes = zz[fs.selected]

interp_J00_jkq = mon_spline_interp(z_nodes, Jm00_1, extrapolate=True)
interp_J20_jkq = mon_spline_interp(z_nodes, Jm20_1, extrapolate=True)

Jm00_interp_jkq = interp_J00_jkq(zz)
Jm20_interp_jkq = interp_J20_jkq(zz)

plt.plot(zz,Jm00_interp_jkq,'-.r')
plt.plot(z_nodes,Jm00_1,'or')
plt.plot(z_nodes,Jm00_initial,'ok')
plt.show()
plt.plot(zz,Jm20_interp_jkq,'-.r')
plt.plot(z_nodes,Jm20_1,'or')
plt.plot(z_nodes,Jm20_initial,'ok')
plt.show()