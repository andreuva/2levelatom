#############################################################################
#                2 LEVEL ATOM ATMOSPHERE SOLVER                             #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz
from tqdm import tqdm
import constants as cte
import parameters as pm
import physical_functions as func


# We define the z0, zl, dz as our heigt grid (just 1D because of a
# plane-parallel atmosfere and axial-simetry)
zz = np.arange(pm.zl, pm.zu, pm.dz)          # compute the 1D grid

# Define the grid in frequencies ( or wavelengths )
ww = np.arange(pm.wl, pm.wu, pm.dw)          # Compute the 1D spectral grid

# Define the directions of the rays
mu = np.linspace(0, 1, pm.nmu)

# ------------------------ INITIAL CONDITIONS -----------------------------
# The initial conditions are the asumption of LTE so:
TT = np.exp(-zz)*1e5
XI = np.exp(-zz)
ww, TT = np.meshgrid(ww, TT)
_, XI = np.meshgrid(ww, XI)

SI0 = func.plank_nu(ww, TT)
I0 = SI0*0
SQ0 = 0*SI


def dIdtau(I, tau, SI):
    dIdtau = I - SI
    return dIdtau


gamma = np.sqrt(2) * pm.a       # a = gamma / (sqrt(2)*sigma)
phy = func.Voigt(ww, 1, gamma, pm.wline)

# WHAT JU, JL, rhoju, rhojl, mu ???
SIL = (2*cte.h*ww/cte.c**2) * np.sqrt((2*Jl+1)/(2*Ju+1)) * \
    (rho00ju + wjujl*(3*mu**2 - 1)/np.sqrt(8)*rho02ju)

SI = phy/(phy+r) * SIL + r/(phy+r) * func.plank_nu(ww, TT)
