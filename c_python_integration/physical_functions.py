#############################################################################
#       DEFINITION THE PHYSICALFUNCTIONS, DISTRIBUTIONS AND PROFILES        #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import scipy.special as special

def gaussian(x, sigma, x0=0):
    """ Return Gaussian line shape at x with a given sigma centred at x0 """
    return 1/(sigma*np.sqrt(2*np.pi)) \
        * np.exp(-((x-x0) / sigma)**2 / 2)


def lorentzian(x, gamma, x0=0):
    """ Return Lorentzian line shape at x with gamma centred at x0"""
    return gamma / np.pi / ((x-x0)**2 + gamma**2)


def voigt(v, a):

    s = abs(v)+a
    d = .195e0*abs(v)-.176e0
    z = a - 1j*v

    if s >= .15e2:
        t = .5641896e0*z/(.5+z*z)
    else:

        if s >= .55e1:

            u = z*z
            t = z*(.1410474e1 + .5641896e0*u)/(.75e0 + u*(.3e1 + u))

        else:

            if a >= d:
                nt = .164955e2 + z*(.2020933e2 + z*(.1196482e2 +
                                    z*(.3778987e1 + .5642236e0*z)))
                dt = .164955e2 + z*(.3882363e2 + z*(.3927121e2 +
                                    z*(.2169274e2 + z*(.6699398e1 + z))))
                t = nt / dt
            else:
                u = z*z
                x = z*(.3618331e5 - u*(.33219905e4 - u*(.1540787e4 - \
                    u*(.2190313e3 - u*(.3576683e2 - u*(.1320522e1 - \
                    .56419e0*u))))))
                y = .320666e5 - u*(.2432284e5 - u*(.9022228e4 - \
                    u*(.2186181e4 - u*(.3642191e3 - u*(.6157037e2 - \
                    u*(.1841439e1 - u))))))
                t = np.exp(u) - x/y
    return t