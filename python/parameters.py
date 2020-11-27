#############################################################################
#            DEFINE SOME OF THE CONSTANTS OF THE PROBLEM                    #
#############################################################################
c = 299792458                   # m/s
h = 6.626070150e-34             # J/s
kb = 1.380649e-23                # J/K
R = 8.31446261815324            # J/K/mol

#############################################################################
#     DEFINITION OF THE PARAMETERS OF THE PROBLEM (THINGS TO CHANGE)        #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
T = 5778                            # T (isotermic) of the medium (not used yet)

zl = -15 #-np.log(1e7)                  # lower boundary optically thick
zu = 9 #-np.log(1e-3)                   # upper boundary optically thin
dz = 0.75                                  # stepsize

w_normaliced = True

if not w_normaliced:
    wl = c/(502e-9)               # lower/upper frequency limit
    wu = c/(498e-9)
    w0 = c/(500e-9)
    wa = w0*(np.sqrt(2*R*T/1e-3))/c          # normalization of the frec. for the Voigt
    dw = (wu-wl)/150                    # points to sample the spectrum
else:
    # parameters for the already normalice wavelengths
    wl = -10
    wu = 10
    w0 = 0
    dw = 0.25

qnd = 14                          # nodes in the gaussian quadrature (# dirs) (odd number)

a = 1e-5                          # dumping Voigt profile a=gam/(2^1/2*sig)
r = 1e-2                          # XCI/XLI
eps = 1e-4                        # Phot. dest. probability (LTE=1,NLTE=1e-4)
dep_col = 0                       # Depolirarization colisions (delta)
Hd = 1                            # Hanle depolarization factor [1/5, 1]

ju = 1
jl = 0

tolerance = 1e-10 #0.5e-3                   # Tolerance for finding the solution
max_iter = 500                            # maximum number of iterations to find the solution

initial_plots = True
plots = True

nn = int((wu-wl)/dw /2)                    # element of frequency at which plot
mm = -1                                    # dir at with plot the maps
