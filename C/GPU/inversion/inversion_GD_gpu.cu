/**********************************************************************/
/*  Program to solve the inversion of the 2 level atom problem        */
/*  Author: Andres Vicente Arevalo      Date: 19-03-2021              */
/**********************************************************************/
extern "C"{
    // Include the needed libraries
    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include <complex.h>
    #include <time.h>
    // include the definition of the parameters of the forward and the inversion
    #include "params.h"

    // Define the grid in heights and wavelengths based on the parameters
    #define nz int((zu-zl)/dz + 1)                /* # number of points in the z axes */
    #define nw int((wu-wl)/dw + 1)                /* # points to sample the spectrum */

    /**************************     "forward_solver.c"      *****************************/
    // Forward solver functions for solving the 2 level atom problem in 1D with the SC method
    // double complex voigt(double v, double a);
    // void psi_calc(double deltaum[], double deltaup[], double psim[], double psio[], double psip[], int mode);
    void RTE_SC_solve(double II[][nw][qnd], double QQ[][nw][qnd], double SI[nz][nw][qnd], double SQ[nz][nw][qnd], double lambda[][nw][qnd], double tau[nz][nw], double mu[qnd]);

    // Actual Forward solver that returns the I and Q of a given parameters
    void solve_profiles(double a,double r, double eps, double dep_col, double Hd, double I_res[nz][nw][qnd], double Q_res[nz][nw][qnd]);

    /**************************   "inversion_routines.c"    ****************************/
    // Some routines that uses random numbers to generate the noise of the "observed" profile
    // double* generate(int n);
    void add_noise_1D(int nn, double array[], double std);

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
}

// Define the "observed" parameters
const double a_sol = 1e-3;                      /* # dumping Voigt profile a=gam/(2^1/2*sig) */
const double r_sol = 1e-3;                     /* # line strength XCI/XLI */
const double eps_sol = 1e-4;                    /* # Phot. dest. probability (LTE=1,NLTE=1e-4) */
const double dep_col_sol = 0.8;                /* # Depolirarization colisions (delta) */
const double Hd_sol = 0.5;                      /* # Hanle depolarization factor [1/5, 1] */

__managed__ double I_full[nz][nw][qnd], Q_full[nz][nw][qnd];
__managed__ double I_0[nw], Q_0[nw], I_1[nw], Q_1[nw], I_obs_sol[nw], Q_obs_sol[nw];
__managed__ double x_0[numpar], x_1[numpar], x_sol[numpar] = {a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol};
__managed__ double x_l[numpar] = {1e-12, 1e-12, 1e-4, 0, 0.2};
__managed__ double x_u[numpar] = {1, 1, 1, 10 ,1};

__managed__ double xs[numpar][numpar], beta[numpar];
__managed__ float progres;

__managed__ double chi2_0, chi2_1, cc=1;

int main(){

    int i, j, k, itt;
    double step_size[numpar] = {1e-2,1e-7,1e-6,1e-1,1e-1};
    int new_point = 1; 

    fprintf(stdout,"=====================================\n");
    fprintf(stdout, "\nSOLUTION PARAMETERS:\n");
    fprintf(stdout, " a = %1.8e\n r = %1.8e\n eps = %1.8e\n delta = %1.8e\n Hd = %1.8e\n\n",\
                          x_sol[0],    x_sol[1],     x_sol[2],      x_sol[3],    x_sol[4]);
    fprintf(stdout,"=====================================\n");

    fprintf(stdout, "Computing the solution profiles... \n\n");
    solve_profiles( x_sol[0], x_sol[1], x_sol[2], x_sol[3], x_sol[4], I_full, Q_full);
    for (j = 0; j < nw; j++){
        I_obs_sol[j] = I_full[nz-1][j][mu_sel];
        Q_obs_sol[j] = Q_full[nz-1][j][mu_sel];
    }
    fprintf(stdout,"=====================================\n");

    add_noise_1D(nw, I_obs_sol, STD);
    add_noise_1D(nw, Q_obs_sol, STD);

    // initialice the parameters to start the inversion
    // for (i = 0; i < numpar; i++){
    //     x_0[i] = (double)rand()/(double)(RAND_MAX) * (x_u[i] - x_l[i]) + x_l[i];
    // }
    x_0[0] = 1e-5; x_0[1] = 1e-5; x_0[2] = 1e-3;  x_0[3] = 1e-1; x_0[4] = 0.6;

    fprintf(stdout, "\nINITIAL PARAMETERS:\n");
    fprintf(stdout, " a = %1.8e\n r = %1.8e\n eps = %1.8e\n delta = %1.8e\n Hd = %1.8e\n\n",\
                          x_0[0],     x_0[1],        x_0[2],        x_0[3],       x_0[4]);
    fprintf(stdout,"=====================================\n");
    
    chi2_0 = chi2_calc(x_0, I_obs_sol, Q_obs_sol, I_0, Q_0, STD, WI, WQ);

    for (itt = 0; itt < max_iter_inversion; itt++){

        fprintf(stdout,"Step %i of %i\n",itt, max_iter_inversion);
        progres =  100.0 * itt/ max_iter_inversion ;
        fprintf(stdout, "[");
        for (i=0; i<70; i++) {
            if ( i < progres*7/10 ) {
                fprintf(stdout, "#");
            }else{
                fprintf(stdout, "-");
            }
        }
        fprintf(stdout, "] %1.2f %%\n", progres);

        if(new_point){
            surroundings_calc(x_0, xs, HH);
            check_range(x_0, xs, x_l, x_u);
            compute_gradient(x_0, xs, beta, I_obs_sol, Q_obs_sol, HH, STD, WI, WQ);
        }

        for (i = 0; i < numpar; i++){
            x_1[i] = x_0[i] - step_size[i]*beta[i]*cc;
        }


        for (i=0; i<numpar; i++){
            if (x_1[i] > x_u[i]){
                x_1[i] = x_u[i];
            }else if (x_1[i] < x_l[i]){
                x_1[i] = x_l[i];
            }
        }
       
        chi2_1 = chi2_calc(x_1, I_obs_sol, Q_obs_sol, I_1, Q_1, STD, WI, WQ);
        fprintf(stdout,"Total Chi^2 %1.6e\n with step parameter of %1.3e\n",chi2_1,cc);
        
        // If we have very good loss stop
        if (chi2_1 < 10){
            break;
        }

        if (chi2_1 <= chi2_0 || itt == 0){
            for (i = 0; i < numpar; i++){
                x_0[i] = x_1[i];
            }
            cc = cc*1.25;
            new_point = 1;
            chi2_0 = chi2_1;

            for (j = 0; j < nw; j++){
                I_0[j] = I_1[j];
                Q_0[j] = Q_1[j];
            }

            fprintf(stdout,"=====================================\n");
            fprintf(stdout, "\nNew parameters:\n");
            fprintf(stdout, " a = %1.8e\n r = %1.8e\n eps = %1.8e\n delta = %1.8e\n Hd = %1.8e\n\n",\
                                x_0[0], x_0[1], x_0[2], x_0[3], x_0[4]);
            fprintf(stdout,"=====================================\n");
        }else{
            cc = cc/2;            
            new_point = 0;
        }
    }
    
    fprintf(stdout, "\nFINISHED AFTER %i ITTERATIONS\n", itt);
    fprintf(stdout, "\n--------------------------------------------------------------------\n");
    fprintf(stdout, "\tINVERTED PARAMETERS\t | \tSOLUTION PARAMETERS\t\n");
    fprintf(stdout, "--------------------------------------------------------------------\n");
    fprintf(stdout, "  a = %1.8e \t\t |  a_sol = %1.8e\n  r = %1.8e \t\t |  r_sol = %1.8e\
    \n  eps = %1.8e \t\t |  eps_sol = %1.8e\n  delta = %1.8e \t |  delta_sol = %1.8e\
    \n  Hd = %1.8e \t\t |  Hd_sol = %1.8e\n",\
    x_0[0], x_sol[0], x_0[1], x_sol[1], x_0[2], x_sol[2], x_0[3], x_sol[3], x_0[4], x_sol[4]);
    fprintf(stdout, "--------------------------------------------------------------------\n");
    
    /* define array, counter, and file name, and open the file */

    FILE *fp;
    fp=fopen("./figures/profiles.txt","w");
    
    fprintf(fp, "# index, I_solution, Q_solution, I_inverted, Q_inverted\n");
    for(i = 0; i < nw; i++){
       fprintf (fp, " %i , %1.8e , %1.8e , %1.8e , %1.8e\n",\
                    i, I_obs_sol[i], Q_obs_sol[i], I_0[i], Q_0[i]);
    }
    fclose (fp);

    fp=fopen("./figures/parameters_inverted.txt","w");

    fprintf(fp, "\nFINISHED AFTER %i ITTERATIONS\n", itt);
    fprintf(fp, "\n------------------------------------------------------------------------\n");
    fprintf(fp, "\tINVERTED PARAMETERS\t | \tSOLUTION PARAMETERS\t\n");
    fprintf(fp, "------------------------------------------------------------------------\n");
    fprintf(fp, "  a = %1.8e \t\t\t |  a_sol = %1.8e\n  r = %1.8e \t\t\t |  r_sol = %1.8e\
    \n  eps = %1.8e \t\t\t |  eps_sol = %1.8e\n  delta = %1.8e \t\t |  delta_sol = %1.8e\
    \n  Hd = %1.8e \t\t\t |  Hd_sol = %1.8e\n",\
    x_0[0], x_sol[0], x_0[1], x_sol[1], x_0[2], x_sol[2], x_0[3], x_sol[3], x_0[4], x_sol[4]);
    fprintf(fp, "------------------------------------------------------------------------\n");
    fprintf(fp, "\t\tEXECUTION PARAMETERS\n");
    fprintf(fp, "------------------------------------------------------------------------\n");
    fprintf(fp, " Maximum iterrations = %i\n Noise of the profiles = %f\
                \n Selected direction of observation = %i\n step of the derivative = %f\
                \n weigths:\t w_I = %f\t w_Q = %f\n",max_iter_inversion,STD,mu_sel,HH,WI,WQ);
    fprintf(fp, "------------------------------------------------------------------------\n");
    fclose (fp);

    return 0;
}
