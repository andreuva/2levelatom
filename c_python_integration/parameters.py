#############################################################################
#     DEFINITION OF THE PARAMETERS OF THE PROBLEM (THINGS TO CHANGE)        #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
zl = -15                                   # lower boundary optically thick
zu = 9                                     # upper boundary optically thin
dz = 0.75                                  # stepsize
nz = int((zu-zl)/dz + 1)

wl = -7
wu = 7
w0 = 0
dw = 0.5
nw = int((wu-wl)/dw + 1)

qnd = 14                          # nodes in the gaussian quadrature (# dirs) (odd number)

ju = 1
jl = 0

tolerance = 1e-10 #0.5e-3               # Tolerance for finding the solution
max_iter = 500                          # maximum number of iterations to find the solution