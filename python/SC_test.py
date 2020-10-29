from main import *
# ------------- TEST ON THE SHORT CHARACTERISTICS METHOD ------------------------
# We define the ilumination just at the bottom boundary
# Initialaice the used tensors
II = np.zeros_like(plank_Ishape)
QQ = np.zeros_like(II)
II[0] = plank_Ishape[0]
# Define the new vectors as the old ones
II_new = np.copy(II)
QQ_new = np.copy(QQ)
# Define the source function as a constant value with the dimensions of II
SI = 0.5*np.ones_like(II)
SQ = 0.25*np.ones_like(QQ)

II_new, QQ_new = RTE_SC_solve(II_new,QQ_new,SI,SQ,zz,mus, 'exp')

II = II*np.exp(-((zz_shape-np.min(zz_shape))/mu_shape)) + 0.5*(1-np.exp(-(zz_shape-np.min(zz_shape)/mu_shape)))
QQ = QQ*np.exp(-((zz_shape-np.min(zz_shape))/mu_shape)) + 0.25*(1-np.exp(-(zz_shape-np.min(zz_shape)/mu_shape)))

plt.plot(zz, II[:, 50, -1], 'b', label='$I$')
plt.plot(zz, QQ[:, 50, -1], 'b--', label='$Q$')
plt.plot(zz, II_new[:, 50, -1], 'rx', label='$I_{calc}$')
plt.plot(zz, QQ_new[:, 50, -1], 'rx', label='$Q_{calc}$')
plt.legend()
plt.show()