import numpy as np
import matplotlib.pyplot as plt

_, I_sol, Q_sol, I_inv, Q_inv, I_initial, Q_initial = np.loadtxt("figures/profiles_jkq.txt",  delimiter=',', unpack=True)

plt.plot(I_sol, 'ok', label=r'$I/B_{\nu}$ "observed"')
plt.plot(I_initial, 'r', label=r'$I/B_{\nu}$ initial parameters')
plt.plot(I_inv, 'b', label=r'$I/B_{\nu}$ inverted parameters')
plt.legend(); plt.xlabel(r'$\nu $')
plt.show()
plt.plot(Q_initial, 'r--', label='$Q/I$ initial parameters')
plt.plot(Q_sol, 'ok', label='$Q/I$ "observed"')
plt.plot(Q_inv, 'b--', label='$Q/I$ inverted parameters')
plt.legend(); plt.xlabel(r'$\nu $')
plt.show()

_ , zz, Jm00, Jm20 = np.loadtxt("figures/radiation_jkq.txt",  delimiter=',', unpack=True)

plt.plot(Jm00, 'r', label=r'$J^0_0$')
plt.plot(Jm20, 'b', label=r'$J^2_0$')
plt.legend(); plt.xlabel(r'$z$')
plt.show()

_, I_sol, Q_sol, I_inv, Q_inv = np.loadtxt("figures/profiles.txt",  delimiter=',', unpack=True)

plt.plot(I_sol, 'ok', label=r'$I/B_{\nu}$ "observed"')
plt.plot(I_inv, 'b', label=r'$I/B_{\nu}$ inverted parameters')
plt.legend(); plt.xlabel(r'$\nu $')
plt.show()
plt.plot(Q_sol, 'ok', label='$Q/I$ "observed"')
plt.plot(Q_inv, 'b--', label='$Q/I$ inverted parameters')
plt.legend(); plt.xlabel(r'$\nu $')
plt.show()