
#ifndef __CUDAEXT__
#define __CUDAEXT__
extern "C"{
    /**********************************************************************/
    /*  Header file of the inversion program                              */
    /*  Author: Andres Vicente Arevalo      Date: 19-03-2021              */
    /**********************************************************************/
    // Include the needed libraries
    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include <complex.h>
    #include <time.h>
    // include the definition of the parameters of the forward and the inversion
    #include "params.h"

    // Define the grid in heights and wavelengths based on the parameters
    #define nz ((zu-zl)/dz + 1)                /* # number of points in the z axes */
    #define nw ((wu-wl)/dw + 1)                /* # points to sample the spectrum */

    void RTE_SC_solve(double II[][nw][qnd], double QQ[][nw][qnd], double SI[nz][nw][qnd], double SQ[nz][nw][qnd], double lambda[][nw][qnd], double tau[nz][nw], double mu[qnd]);
}
#endif
