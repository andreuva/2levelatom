#############################################################################
#     DEFINITION OF THE PARAMETERS OF THE PROBLEM (THINGS TO CHANGE)        #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import constants as cte
T = 5778                            # T (isotermic) of the medium (not used yet)

zl = -15 #-np.log(1e7)                  # lower boundary optically thick
zu = 9 #-np.log(1e-3)                   # upper boundary optically thin
dz = 1                                  # stepsize

w_normaliced = True

if not w_normaliced:
    wl = cte.c/(502e-9)               # lower/upper frequency limit
    wu = cte.c/(498e-9)
    w0 = cte.c/(500e-9)
    wa = w0*(np.sqrt(2*cte.R*T/1e-3))/cte.c          # normalization of the frec. for the Voigt
    dw = (wu-wl)/150                    # points to sample the spectrum
else:
    # parameters for the already normalice wavelengths
    wl = -5
    wu = 5
    w0 = 0
    dw = 0.5

gaussian = True
qnd = 8                            # nodes in the gaussian quadrature (# dirs) (odd number)

a = 1e-3                            # dumping Voigt profile a=gam/(2^1/2*sig)
r = 1e-12                           # XCI/XLI
eps = 1e-4                          # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col = 0                       # Depolirarization colisions (delta)
Hd = 1                            # Hanle depolarization factor [1/5, 1]
ju = 1
jl = 0

tolerance = 1e-5 #0.5e-3                   # Tolerance for finding the solution
max_iter = 10000                            # maximum number of iterations to find the solution
initial_plots = False
plots = True
nn = int((wu-wl)/dw /2)                    # element of frequency at which plot
mm = -1                                    # dir at with plot the maps