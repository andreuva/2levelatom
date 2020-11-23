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

a_sol = 1e-2      #1e-5                    # dumping Voigt profile a=gam/(2^1/2*sig)
r_sol = 1e-5      #1                    # XCI/XLI
eps_sol = 1e-4                        # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col_sol = 0                       # Depolirarization colisions (delta)
Hd_sol = 1                            # Hanle depolarization factor [1/5, 1]

print(" Solution parameters: ")
print(f" a = {a_sol}\n r = {r_sol}\n eps = {eps_sol}\n delta = {dep_col_sol}\n Hd = {Hd_sol}\n")
print("Computing the solution profiles:")
I_sol, Q_sol = fs.solve_profiles(a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol)

random.seed(12455)
a = random.uniform(1e-10,1)
r = random.uniform(1e-15,1)
eps = random.uniform(1e-4,1)
dep_col = random.uniform(0,10)
Hd = random.uniform(1/5, 1)
print("\n Initial parameters: ")
print(f" a = {a}\n r = {r}\n eps = {eps}\n delta = {dep_col}\n Hd = {Hd}\n")
print("Computing the initial profiles:")
I_initial, Q_initial = fs.solve_profiles(a, r, eps, dep_col, Hd)


def chi2(params):
    a, r, eps, dep_col, Hd = params
    # print("\n New parameters: ")
    # print(f" a = {a}\n r = {r}\n eps = {eps}\n delta = {dep_col}\n Hd = {Hd}\n")
    print("\n Computing the new profiles:")
    I,Q = fs.solve_profiles(a, r, eps, dep_col, Hd)
    chi2 = np.sum((I-I_sol)**2/(I_sol + 1e-200) + (Q-Q_sol)**2/(Q_sol + 1e-200))
    print(f'Chi^2 of this profiles is: {np.max(np.abs(chi2))} ' )
    return chi2

x0 = np.array([a,r,eps,dep_col,Hd])
xl = [1e-4,1e-12,1e-4,0,0.2]
xu = [1,1,1,10,1]
bounds = Bounds(xl,xu)
result = opt.minimize(chi2,x0, bounds=bounds)
print("\n Solution Parameters: ")
print(f" a = {result.x[0]}\n r = {result.x[1]}\n eps = {result.x[2]}\n delta = {result.x[3]}\n Hd = {result.x[4]}\n")

I_res, Q_res = fs.solve_profiles(result.x[0], result.x[1], result.x[2], result.x[3], result.x[4])

plt.plot((I_sol)[-1, :, -1], 'k', label=r'$I/B_{\nu}$ "observed"')
plt.plot((I_initial)[-1, :, -1], 'r', label=r'$I/B_{\nu}$ initial')
plt.plot((I_res)[-1, :, -1], 'b', label=r'$I/B_{\nu}$ inverted')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()
plt.plot((Q_initial/I_initial)[-1, :, -1], 'r--', label='$Q/I$ initial')
plt.plot((Q_sol/I_sol)[-1, :, -1], 'k--', label='$Q/I$ "observed"')
plt.plot((Q_res/I_res)[-1, :, -1], 'b--', label='$Q/I$ inverted')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()