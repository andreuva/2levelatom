from ctypes import c_void_p ,c_double, c_int, c_float, cdll
import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import time
import forward_solver_py as fs

a = 1e-6      #1e-5,1e-2 ,1e-4                # dumping Voigt profile a=gam/(2^1/2*sig)
r = 1e-8      #1,1e-4   ,1e-10                 # XCI/XLI
eps = 1e-2                          # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col = 0.5             #0.1          # Depolirarization colisions (delta)
Hd = 1                  #1          # Hanle depolarization factor [1/5, 1]

zl = -15
zu = 9
dz = 0.75
nz = int((zu-zl)/dz + 1)

wl = -10
wu = 10
dw = 0.25
nw = int((wu-wl)/dw + 1)

qnd = 14

time1 = time.time()
I_py, Q_py = fs.solve_profiles(a, r, eps, dep_col, Hd)
time2 = time.time() - time1
print("py running time in seconds:", time2)
py_time = time2

lib = cdll.LoadLibrary("/home/andreuva/Documents/2 level atom/2levelatom/python/c_python_integration/forward_solver.so")
solve_profiles = lib.solve_profiles
solve_profiles.restype = ndpointer(dtype=c_double , shape=(nz*nw*qnd*2,))

time1 = time.time()
result = solve_profiles(c_float(a), c_float(r), c_float(eps), c_float(dep_col), c_float(Hd))
I_c = result[:nz*nw*qnd].reshape(I_py.shape)
Q_c = result[nz*nw*qnd:].reshape(Q_py.shape)
time2 = time.time() - time1
print("c  running time in seconds:", time2)

c_time = time2
print("speedup:", py_time/c_time )

import matplotlib.pyplot as plt

plt.plot((I_py)[-1, :, -1], 'r', label=r'$I/B_{\nu}$ python')
plt.plot((I_c)[-1, :, -1], 'b', label=r'$I/B_{\nu}$ C')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()
plt.plot((Q_py)[-1, :, -1], 'r', label='$Q$ python')
plt.plot((Q_c)[-1, :, -1], 'b', label='$Q$ C')
plt.legend(); plt.xlabel(r'$\nu\ (Hz)$')
plt.show()