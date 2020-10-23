#############################################################################
#     DEFINITION OF THE PARAMETERS OF THE PROBLEM (THINGS TO CHANGE)        #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import constants as cte

zl = -np.log(1e0)                    # lower boundary optically thick
zu = -np.log(1e-3)                   # upper boundary optically thin
dz = (zu-zl)/200                     # stepsize

wl = cte.c/(5000e-9)               # lower/upper frequency limit
wu = cte.c/(350e-9)
w0 = cte.c/(1000e-9)
wa = cte.c/(1000e-9) * 0.05          # normalization of the frec. for the Voigt
dw = (wu-wl)/250                    # 100 points to sample the spectrum

qnd = 100                            # nodes in the gaussian quadrature (# dirs)

T = 5778                            # T (isotermic) of the medium

a = 0.2                               # dumping Voigt profile a=gam/(2^1/2*sig)
r = 0.5                             # line strength XCI/XLI
eps = 1e-4                          # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col = 0.5                       # Depolirarization colisions (delta)
Hd = 1/5                            # Hanle depolarization factor [1/5, 1]
ju = 1
jl = 0

tolerance = 1e-15                         # Tolerance for finding the solution
max_iter = 20                            # maximum number of iterations to find the solution