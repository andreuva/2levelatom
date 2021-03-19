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
const static short nz = (zu-zl)/dz + 1;                /* # number of points in the z axes */
const static short nw = (wu-wl)/dw + 1;                /* # points to sample the spectrum */

/**************************     "forward_solver.c"      *****************************/
// Forward solver functions for solving the 2 level atom problem in 1D with the SC method
double complex voigt(double v, double a);
void psi_calc(double deltaum[], double deltaup[], double psim[], double psio[], double psip[], int mode);
void RTE_SC_solve(double II[][nw][qnd], double QQ[][nw][qnd], double SI[nz][nw][qnd], double SQ[nz][nw][qnd], double lambda[][nw][qnd], double tau[nz][nw], double mu[qnd]);

// Actual Forward solver that returns the I and Q of a given parameters
void solve_profiles(double a,double r, double eps, double dep_col, double Hd, double I_res[nz][nw][qnd], double Q_res[nz][nw][qnd]);

/**************************   "inversion_routines.c"    ****************************/
// Some routines that uses random numbers to generate the noise of the "observed" profile
double* generate(int n);
void add_noise_1D(int nn, double array[nn], double std);

// Calculations of chi2 and implicitly the profiles
double chi2_calc(double params[numpar], double I_obs_sol[nw], double Q_obs_sol[nw], double I_obs[nw], double Q_obs[nw], double std, double w_I, double w_Q);
// compute the neighbor points of a given parameters
void surroundings_calc(double x_0[numpar], double surroundings[numpar][numpar], double hh);
// check that the range of the parameters is in the valid range
void check_range(double x_0[], double xs[][numpar], double x_l[], double x_u[]);
// compute the gradient of a given parameters
void compute_gradient(double params[numpar], double surroundings[numpar][numpar], double beta[numpar], double I_obs_sol[nw], double Q_obs_sol[nw], double hh, double std, double w_I, double w_Q);

/**************************   "integrals_quadratures.c"    ****************************/
// integrals and quadrature functions
void gauleg(double x1, double x2, double x[], double w[], int n);
double num_gaus_quad(double y[], double weigths[], int nn);
double trapezoidal(double yy[], double xx[], int nn);