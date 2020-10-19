#############################################################################
#     DEFINITION OF THE PARAMETERS OF THE PROBLEM (THINGS TO CHANGE)        #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import constants as cte

zl = -np.log(1e3)                   # lower boundary optically thick
zu = -np.log(1e-3)                  # upper boundary optically thin
dz = 1e-1                           # stepsize

wl = cte.c/(600*1e-9)               # lower/upper frequency limit
wu = cte.c/(500*1e-9)
dw = (wu-wl)/100                    # 100 points to sample the spectrum

qnd = 20                            # nodes in the gaussian quadrature (# dirs)

T = 5778                            # T (isotermic) of the medium

a = 1                               # dumping Voigt profile a=gam/(2^1/2*sig)
r = 0.5                             # line strength XCI/XLI
eps = 1                             # Phot. dest. probability (LTE=1,NLTE=12-4)
dep_col = 0.5                       # Depolirarization colisions (delta)
Hd = 1/5                            # Hanle depolarization factor [1/5, 1]
ju = 1
jl = 0