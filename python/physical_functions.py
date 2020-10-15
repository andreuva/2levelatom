#############################################################################
#       DEFINITION THE PHYSICALFUNCTIONS, DISTRIBUTIONS AND PROFILES        #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import constants as cte


def Gaussian(x, sigma, x0=0):
    """ Return Gaussian line shape at x with a given sigma centred at x0 """
    return 1/(sigma*np.sqrt(2*np.pi)) \
        * np.exp(-((x-x0) / sigma)**2 / 2)


def Lorentzian(x, gamma, x0=0):
    """ Return Lorentzian line shape at x with gamma centred at x0"""
    return gamma / np.pi / ((x-x0)**2 + gamma**2)


def Voigt(x, sigma, gamma, x0=0):
    """
    Return the Voigt line shape at x with Lorentzian component gamma
    and Gaussian component sigma.
    """
    return np.real(wofz(((x-x0) + 1j*gamma)/sigma/np.sqrt(2))) \
        / sigma / np.sqrt(2*np.pi)


def plank_nu(nu, T):
    """
    Return the Plank function at a given temperature and frequency (IS)
    """
    return (2*cte.h*nu**3)/(cte.c**2) * (1/(np.exp(cte.h*nu/(cte.kb*T)) - 1))


def plank_lamb(lamb, T):
    """
    Return the Plank function at a given temperature and wavelength (IS)
    """
    return (2*cte.h*cte.c**2)/(lamb**5) * \
        (1/(np.exp(cte.h*cte.c/(lamb*cte.kb*T)) - 1))
