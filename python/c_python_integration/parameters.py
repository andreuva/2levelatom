#############################################################################
#     DEFINITION OF THE PARAMETERS OF THE PROBLEM (THINGS TO CHANGE)        #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import constants as cte
T = 5778                            # T (isotermic) of the medium (not used yet)

zl = -15 #-np.log(1e7)                  # lower boundary optically thick
zu = 9 #-np.log(1e-3)                   # upper boundary optically thin
dz = 0.75                                  # stepsize

w_normaliced = True

if not w_normaliced:
    wl = cte.c/(502e-9)               # lower/upper frequency limit
    wu = cte.c/(498e-9)
    w0 = cte.c/(500e-9)
    wa = w0*(np.sqrt(2*cte.R*T/1e-3))/cte.c          # normalization of the frec. for the Voigt
    dw = (wu-wl)/150                    # points to sample the spectrum
else:
    # parameters for the already normalice wavelengths
    wl = -10
    wu = 10
    w0 = 0
    dw = 0.25

qnd = 14                          # nodes in the gaussian quadrature (# dirs) (odd number)

ju = 1
jl = 0

tolerance = 1e-10 #0.5e-3                   # Tolerance for finding the solution
max_iter = 500                            # maximum number of iterations to find the solution                                  # dir at with plot the maps