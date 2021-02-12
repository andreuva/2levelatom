import forward_solver_jkq  as fs
import forward_solver_py_J as sfs
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator as mon_spline_interp
import parameters as pm


a_sol = 1e-6      #1e-3                 # dumping Voigt profile a=gam/(2^1/2*sig)
r_sol = 1e-8      #1e-2                 # XCI/XLI
eps_sol = 1e-3    #1e-1                 # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col_sol = 2   #8                    # Depolirarization colisions (delta)
Hd_sol = .4       #0.8                  # Hanle depolarization factor [1/5, 1]
mu = 9 #int(fs.pm.qnd/2)

zz = np.arange(pm.zl, pm.zu + pm.dz, pm.dz)          # compute the 1D grid
z_nodes = zz[fs.selected]

I_sol, Q_sol, Jm00_sol, Jm20_sol = sfs.solve_profiles(a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol)
Jm00_sol, Jm20_sol = Jm00_sol[fs.selected], Jm20_sol[fs.selected]
I_sol = I_sol[-1,:,mu].copy()
Q_sol = Q_sol[-1,:,mu].copy()

I, Q, Jm00s, Jm20s = fs.solve_profiles( a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol, Jm00_sol, Jm20_sol )
# for i in range(1000):
I, Q, Jm00s, Jm20s = fs.solve_profiles( a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol, Jm00s, Jm20s )
Is = I[-1,:,mu].copy()
Qs = Q[-1,:,mu].copy()

interp_J00_sol = mon_spline_interp(z_nodes, Jm00_sol, extrapolate=True)
interp_J20_sol = mon_spline_interp(z_nodes, Jm20_sol, extrapolate=True)
interp_J00 = mon_spline_interp(z_nodes, Jm00s, extrapolate=True)
interp_J20 = mon_spline_interp(z_nodes, Jm20s, extrapolate=True)

Jm00_interp_sol = interp_J00_sol(zz)
Jm20_interp_sol = interp_J20_sol(zz)
Jm00_interp = interp_J00(zz)
Jm20_interp = interp_J20(zz)

plt.plot(zz,Jm00_interp_sol,'-.b', label='Solution')
plt.plot(z_nodes,Jm00_sol,'ob')
plt.plot(zz,Jm00_interp,'-.r')
plt.plot(z_nodes,Jm00s,'or')
plt.show()
plt.plot(zz,Jm20_interp_sol,'-.b', label='Solution')
plt.plot(z_nodes,Jm20_sol,'ob')
plt.plot(zz,Jm20_interp,'-.r')
plt.plot(z_nodes,Jm20s,'or')
plt.show()

plt.plot(I_sol, 'r--', label=r'$I/B_{\nu}$ "observed"')
plt.plot(Is, 'b--', label=r'$I/B_{\nu}$ initial')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()
plt.plot(Q_sol, 'r--', label='$Q/I$ initial')
plt.plot(Qs, 'b--', label='$Q/I$ "observed"')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()
