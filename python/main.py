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

# ------------------------ INITIAL CONDITIONS -----------------------------
# The initial conditions are the asumption of LTE so:
TT = np.exp(-zz)*1e5
XI = np.exp(-zz)
ww, TT = np.meshgrid(ww, TT)
SI = func.plank_nu(ww, TT)
SQ = 0*SI
