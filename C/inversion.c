/***************************************************************
*        2 LEVEL ATOM ATMOSPHERE SOLVER                        *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
****************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include "integratives.h"
#include "params.h"
#include "subroutines.c"
#include "forward_solver.c"

const int mu_sel = 9;
const float h = 1e-8;
const int numpar = 5;
const float w_I = 1e-1, w_Q = 1e2;
const int max_iter_inversion = 100;

double* generate(int n)
{
    int i;
    int m = n + n % 2;
    double* values = (double*)calloc(m,sizeof(double));
    double average, deviation;
 
    if ( values )
    {
        for ( i = 0; i < m; i += 2 )
        {
            double x,y,rsq,f;
            do {
                x = 2.0 * rand() / (double)RAND_MAX - 1.0;
                y = 2.0 * rand() / (double)RAND_MAX - 1.0;
                rsq = x * x + y * y;
            }while( rsq >= 1. || rsq == 0. );
            f = sqrt( -2.0 * log(rsq) / rsq );
            values[i]   = x * f;
            values[i+1] = y * f;
        }
    }
    return values;
}

void add_noise_1D(int nn, double array[nn], double std, double S_N){
    
    int i;
    double max=-1e50, min=1e50;
    double* normal_noise;

    // initialice the random number generator
    srand((unsigned int)time(NULL));
    normal_noise = generate(nn);

    if (S_N > 0){
        for (i = 0; i < nn; i++)
        {
            if ( array[i] > max ){
                max = array[i];
            }
            if ( array[i] < min ){
                min = array[i];
            } 
        }
        std = sqrt((max-min)*(max-min)/(S_N*S_N));   /* Compute the std based on the S_N */
    }

    for (i = 0; i < nn; i++){
        normal_noise[i] = normal_noise[i] * std;
        array[i] = array[i] + normal_noise[i];
    }
    return;
}

void chi2_calc(float params[5], double I_obs_sol[nw], double Q_obs_sol[nw],
double I_obs[nw], double Q_obs[nw], float std, float w_I, float w_Q){

    double II[nz][nw][qnd], QQ[nz][nw][qnd];
    double chi2 = 0.;

    solve_profiles(params[0], params[1], params[2], params[3], params[4], II, QQ);
    for (int j = 0; j < nw; j++){
        I_obs[j] = II[nz-1][j][mu_sel];
        Q_obs[j] = QQ[nz-1][j][mu_sel];
    }
    
    for (int j = 0; j < nw; j++)
    {
        chi2 = chi2 + ( w_I*(I_obs[j] - I_obs_sol[j])*(I_obs[j] - I_obs_sol[j]) \
        + w_Q*(Q_obs[j]-Q_obs_sol[j])*(Q_obs[j]-Q_obs_sol[j]) )/(std*std*2*nw);
    }
    fprintf(stdout,"Chi^2 of this profiles is: %1.5e", chi2 );

    return;
}

void surroundings_calc(float x_0[5], float surroundings[5][5], float h){

    float delta[5];
    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){ delta[j] = 0;}
        delta[i] = h;
        for(int j = 0; j < 5; j++){ surroundings[i][j] = x_0[j] + delta[j];}
    }

    return;
}

int main(){

    double I_sol[nz][nw][qnd], Q_sol[nz][nw][qnd];
    double I_obs[nw], Q_obs[nw];
    float x_l[5] = {1e-12, 1e-12, 1e-4, 0, 0.2};
    float x_u[5] = {1, 1, 1, 10 ,1};
    float x_0[5], x_sol[5] = {a, r, eps, dep_col, Hd};
    float std = 1e-10;
    int i, j, k;

    solve_profiles( x_sol[0], x_sol[1], x_sol[2], x_sol[3], x_sol[4], I_sol, Q_sol);
    // add_noise(I_sol, std_I);
    // add_noise(Q_sol, std_Q);

    for (j = 0; j < nw; j++){
        I_obs[j] = I_sol[nz-1][j][mu_sel];
        Q_obs[j] = Q_sol[nz-1][j][mu_sel];
    }

    add_noise_1D(nw, I_obs, std, -1.);
    add_noise_1D(nw, Q_obs, std, -1.);

    // initialice the parameters to start the inversion
    for (i = 0; i < 5; i++){
        x_0[i] = (float)rand()/(float)(RAND_MAX) * (x_u[i] - x_l[i]) + x_l[i];
    }
    
    double I_0[nw], Q_0[nw];

    chi2_calc(x_0, I_obs, Q_obs, I_0, Q_0, std, w_I, w_Q);

    float lambd = 1e-5;
    int new_point = 1;
    float xs[5][5];

    for (int itt = 0; itt < max_iter_inversion; itt++){
        fprintf(stdout, "itteration %i\t with a lambda of: %1.2e\n",itt,lambd);

        if(new_point){
            fprintf(stdout, "\nNew parameters:\n");
            fprintf(stdout, " a = %1.8e\n r = %1.8e\n eps = %1.8e\n delta = %1.8e\n Hd = %1.8e\n",\
            x_0[1], x_0[1], x_0[2], x_0[3], x_0[4]);
            surroundings_calc(x_0, xs, h);
            // compute_alpha_beta(alpha, beta, I_sol, Q_sol, I_0, Q_0, x_0, xs, std, w_I, w_Q);
        }


    }
    return 0;
}
