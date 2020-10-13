#############################################################################
#                2 LEVEL ATOM ATMOSPHERE SOLVER                             #
#                AUTHOR: ANDRES VICENTE AREVALO                             #
#############################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz
from tqdm import tqdm
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


# We define the z0, zl, dz as our heigt grid (just 1D because of a
# plane-parallel atmosfere and axial-simetry)
zl = -np.log(1e3)                   # lower boundary optically thick
zu = -np.log(1e-3)                  # upper boundary optically thin
dz = 1e-1                           # define the stepsize
zz = np.arange(zl, zu, dz)          # compute the 1D grid

# Define the grid in frequencies ( or wavelengths )
wl = cte.c/(600*1e-9)           # Define the lower and upper frequency limit
wu = cte.c/(500*1e-9)
dw = (wu-wl)/100                # Define 100 points to sample the spectrum
ww = np.arange(wl, wu, dw)

'''# Check for the plank function to work properly
xw = np.linspace(1e3, 1e15, 1e4)
# xl = np.linspace(100e-9, 1500e-9, 1e4)
yw = plank_nu(xw, 5778)
# yl = plank_lamb(xl, 5778)
plt.plot(xw, yw)
plt.xlabel('frequency (Hz)')
plt.ylabel('Spectral irradiance $W s sr^{-1} m^{-2}$')
plt.title('Blackbody spectrum 5778K')
plt.show()
# plt.plot(xl, yl)
# plt.show() '''

# ------------------------ INITIAL CONDITIONS -----------------------------
