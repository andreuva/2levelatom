#############################################################################
#     DEFINITION OF THE PARAMETERS OF THE PROBLEM (THINGS TO CHANGE)        #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
zl = -15                                   # lower boundary optically thick
zu = 9                                     # upper boundary optically thin
dz = 0.75                                  # stepsize
nz = int((zu-zl)/dz + 1)                   # number of points in the grid

wl = -6                                    # lower boundary in wavelength (normalized)
wu = 6                                     # upper boundary in wavelength (normalized)
w0 = 0                                     # line position in wavelength (normalized)
dw = 0.25                                  # step size in the wavelengths grid (normalized)
nw = int((wu-wl)/dw + 1)                   # number of points in the grid

qnd = 14                                   # nodes in the gaussian quadrature (# dirs) (odd number)

ju = 1                                     # Atomic numbers in the upper and lower level of the atom
jl = 0

tolerance = 1e-10 #0.5e-3                  # Tolerance for finding the solution in the forward solver
max_iter = 500                             # maximum number of iterations to find the solution in the forward solver

dump_lev_marq = 1.5                        # dumping parameter of the lev-marq (Deprecated)

nodes_sep = 10                             # distance (in points) at wich to grab the nodes

#############################################################################
#            DEFINE SOME OF THE CONSTANTS OF THE PROBLEM                    #
#############################################################################
c = 299792458                   # m/s
h = 6.626070150e-34             # J/s
kb = 1.380649e-23               # J/K
R = 8.31446261815324            # J/K/mol
T = 5778                        # T (isotermic) of the medium