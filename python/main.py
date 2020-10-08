#############################################################################
#                2 LEVEL ATOM ATMOSPHERE SOLVER                             #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz
from tqdm import tqdm


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


# Define the constants of the problem:
c = 299792458                   # m/s

# We define the z0, zl, dz as our heigt grid (just 1D because of a
# plane-parallel atmosfere and axial-simetry)
zl = -np.log(1e3)                   # lower boundary optically thick
zu = -np.log(1e-3)                  # upper boundary optically thin
dz = 1e-1                           # define the stepsize
zz = np.arange(zl, zu, dz)          # compute the 1D grid

# Define the grid in frequencies ( or wavelengths )
wl = c/(600*1e-9)           # Define the lower and upper frequency limit
wu = c/(500*1e-9)
dw = (wu-wl)/100            # Define 100 points to sample the spectrum
ww = np.arange(wl, wu, dw)
