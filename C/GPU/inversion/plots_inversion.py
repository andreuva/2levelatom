import numpy as np
import matplotlib.pyplot as plt

_, I_sol, Q_sol, I_inv, Q_inv = np.loadtxt("../figures/profiles.txt",  delimiter=',', unpack=True)

plt.plot(I_sol, 'ok', label=r'$I/B_{\nu}$ "observed"')
# plt.plot(I_initial, 'r', label=r'$I/B_{\nu}$ initial parameters')
plt.plot(I_inv, 'b', label=r'$I/B_{\nu}$ inverted parameters')
plt.legend(); plt.xlabel(r'$\nu $')
plt.show()
# plt.plot(Q_initial, 'r--', label='$Q/I$ initial parameters')
plt.plot(Q_sol, 'ok', label='$Q/I$ "observed"')
plt.plot(Q_inv, 'b--', label='$Q/I$ inverted parameters')
plt.legend(); plt.xlabel(r'$\nu $')
plt.show()