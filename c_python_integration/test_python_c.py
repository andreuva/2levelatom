from ctypes import c_void_p ,c_double, c_int, c_float, cdll
from numpy.ctypeslib import ndpointer
import numpy as np
import time
import forward_solver_py as fs
import parameters as pm

nz = int((pm.zu-pm.zl)/pm.dz + 1)
nw = int((pm.wu-pm.wl)/pm.dw + 1)

time1 = time.time()

I_py, Q_py = fs.solve_profiles(pm.a, pm.r, pm.eps, pm.dep_col, pm.Hd)

time2 = time.time() - time1
print("py running time in seconds:", time2)
py_time = time2

lib = cdll.LoadLibrary("/home/andreuva/Documents/2 level atom/2levelatom/c_python_integration/forward_solver.so")
solve_profiles = lib.solve_profiles
solve_profiles.restype = ndpointer(dtype=c_double , shape=(nz*nw*pm.qnd*2,))

time1 = time.time()

result = solve_profiles(c_float(pm.a), c_float(pm.r), c_float(pm.eps), c_float(pm.dep_col), c_float(pm.Hd))
I_c = result[:nz*nw*pm.qnd].reshape(I_py.shape)
Q_c = result[nz*nw*pm.qnd:].reshape(Q_py.shape)

time2 = time.time() - time1
print("c  running time in seconds:", time2)
c_time = time2

print("speedup:", py_time/c_time )

a_pos = np.logspace(-12,0, num=10)
r_pos = np.logspace(-12,0, num=10)
eps_pos = np.logspace(-4,0, num=6)
dep_col_pos = np.logspace(-4,1, num=7)
Hd_pos = np.logspace(-0.68,0, num=6)

Idif = []
Qdif = []
Imax = []
Qmax = []
Imin = []
Qmin = []
errors = 0

for a in a_pos:
    for r in r_pos:
        for eps in eps_pos:
            for dep_col in dep_col_pos:
                for Hd in Hd_pos:
                    I_py, Q_py = fs.solve_profiles(a, r, eps, dep_col, Hd)
                    result = solve_profiles(c_float(a), c_float(r), c_float(eps), c_float(dep_col), c_float(Hd))
                    I_c = result[:nz*nw*pm.qnd].reshape(I_py.shape)
                    Q_c = result[nz*nw*pm.qnd:].reshape(Q_py.shape)
                    if (np.min(I_c) < -1e-4 or np.min(I_py) < -1e-4): errors = errors +1
                    Idif.append(np.max(np.abs(I_c - I_py)))
                    Qdif.append(np.max(np.abs(Q_c - Q_py)))
                    Imax.append(np.max(I_py))
                    Qmax.append(np.max(Q_py))
                    Imin.append(np.min(I_py))
                    Qmin.append(np.min(Q_py))
                    print(a,r,eps,dep_col,Hd, "\t\t", errors)


with open('test_c_python.npy', 'wb') as f:
    np.save(f, Idif)
    np.save(f, Qdif)
    np.save(f, Imax)
    np.save(f, Qmax)
    np.save(f, Imin)
    np.save(f, Qmin)

import matplotlib.pyplot as plt

plt.plot((I_py)[-1, :, -1], 'r', label=r'$I/B_{\nu}$ python')
plt.plot((I_c)[-1, :, -1], 'b', label=r'$I/B_{\nu}$ C')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()
plt.plot((Q_py)[-1, :, -1], 'r', label='$Q$ python')
plt.plot((Q_c)[-1, :, -1], 'b', label='$Q$ C')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()
