/***************************************************************
*        2 LEVEL ATOM ATMOSPHERE SOLVER                        *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
****************************************************************/

#include <stdio.h>
#include <math.h>

int main() {
    const float zl = 1e0;                    /* optical thicknes of the lower boundary */
    const float zu = 1e-3;                   /* optical thicknes of the upper boundary */
    const short dz = 200;                    /* # number of step in the z axes */

    const float wl = 5000e-9;                /* # lower/upper frequency limit (lambda in nm) */
    const float wu = 350e-9;
    const float w0 = 1000e-9;
    const short dw = 250;                    /* # points to sample the spectrum */

    const short qnd = 100;                   /* # nodes in the gaussian quadrature (# dirs) */

    const int T = 5778;                    /* # T (isotermic) of the medium */

    const float a = 10;                      /* # dumping Voigt profile a=gam/(2^1/2*sig) */
    const float r = 0.5;                     /* # line strength XCI/XLI */
    const float eps = 1e-1;                  /* # Phot. dest. probability (LTE=1,NLTE=1e-4) */
    const float dep_col = 0.5;               /* # Depolirarization colisions (delta) */
    const float Hd = 0.2;                    /* # Hanle depolarization factor [1/5, 1] */
    const short ju = 1;
    const short jl = 0;

    const float tolerance = 1e-15;           /* # Tolerance for finding the solution */

    /*  DEFINE SOME OF THE CONSTANTS OF THE PROBLEM  */
    const int c = 299792458;                   /* # m/s */
    const float h = 6.626070150e-34;             /* # J/s */
    const float kb = 1.380649e-23;                /* # J/K */

    fprintf(stdout, "Depolarization colisions:   %e \n", dep_col);
    return 0;
}