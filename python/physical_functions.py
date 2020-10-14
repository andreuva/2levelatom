#############################################################################
#       DEFINITION THE PHYSICALFUNCTIONS, DISTRIBUTIONS AND PROFILES        #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import numpy as np
import constants as cte


def Gaussian(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha \
        * np.exp(-(x / alpha)**2 * np.log(2))


def Lorentzian(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x**2 + gamma**2)


def Voigt(x, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.
    """
    sigma = alpha / np.sqrt(2 * np.log(2))
    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) \
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
