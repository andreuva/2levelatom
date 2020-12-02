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

#define numpar 5

const int mu_sel = 9;
const double h = 1e-8;
const double w_I = 1e-1, w_Q = 1e2;
const int max_iter_inversion = 100;


double* generate(int n)
{
    int i;
    int m = n + n % 2;
    double* values = (double*)calloc(m,sizeof(double));
    double average, deviation;
 
    if ( values ){
        for ( i = 0; i < m; i += 2 ){
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
        for (i = 0; i < nn; i++){
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

double chi2_calc(double params[numpar], double I_obs_sol[nw], double Q_obs_sol[nw],
double I_obs[nw], double Q_obs[nw], float std, double w_I, double w_Q){

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
    fprintf(stdout,"Chi^2 of this profiles is: %1.5e\n", chi2 );

    return chi2;
}

void surroundings_calc(double x_0[numpar], double surroundings[numpar][numpar], double h){

    double delta[numpar];
    int i,j;
    
    for(i = 0; i < numpar; i++){
        for(j = 0; j < numpar; j++){ delta[j] = 0; }
        delta[i] = h;
        for(j = 0; j < numpar; j++){ surroundings[i][j] = x_0[j] + delta[j]; }
    }

    return;
}


void compute_alpha_beta(double alpha[numpar][numpar], double beta[numpar], double I_sol[nw], double Q_sol[nw],\
double I_0[nw],double Q_0[nw], double x_0[numpar], double xs[numpar][numpar], double std, double w_I,double w_Q){

    double I_p[nz][nw][qnd], Q_p[nz][nw][qnd], II[nw], QQ[nw], Is[numpar][nw], Qs[numpar][nw];
    double dQ[numpar][nw], dI[numpar][nw];
    int i, j, l;

    for (i = 0; i < numpar; i++){
        solve_profiles( xs[i][0], xs[i][1], xs[i][2], xs[i][3], xs[i][4], I_p, Q_p);

        for (j = 0; j < nw; j++){
            Is[i][j] = I_p[nz-1][j][mu_sel];
            Qs[i][j] = Q_p[nz-1][j][mu_sel];

            dI[i][j] = (I_0[j] - Is[i][j])/h;
            dQ[i][j] = (Q_0[j] - Qs[i][j])/h;
        }
    }
    
    for (i = 0; i < numpar; i++){
        beta[i] = 0;
        for (l = 0; l < nw; l++){
            beta[i] = beta[i] + (w_I*(Is[i][l] - I_sol[l])*dI[i][l] + w_Q*(Qs[i][l] - Q_sol[l])*dQ[i][l])/(nw*std*std);
        }
        
        for (j = 0; j < numpar; j++){
            alpha[i][j] = 0;
            for (l = 0; l < nw; l++){
                alpha[i][j] = alpha[i][j] + (w_I*dI[i][l]*dI[j][l] + w_Q*dQ[i][l]*dQ[j][l])/nw;
            }
        }
    }
    
    return;
}




int main(){

    double I_sol[nz][nw][qnd], Q_sol[nz][nw][qnd];
    double I_obs[nw], Q_obs[nw];
    double x_l[numpar] = {1e-12, 1e-12, 1e-4, 0, 0.2};
    double x_u[numpar] = {1, 1, 1, 10 ,1};
    double x_0[numpar], x_1[numpar], x_sol[numpar] = {a, r, eps, dep_col, Hd};
    double std = 1e-10;
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
    for (i = 0; i < numpar; i++){
        x_0[i] = (float)rand()/(float)(RAND_MAX) * (x_u[i] - x_l[i]) + x_l[i];
    }
    
    double I_0[nw], Q_0[nw], I_1[nw], Q_1[nw];
    double chi2_0, chi2_1;
    double lambd = 1e-5;
    int new_point = 1;
    double xs[numpar][numpar];
    double alpha[numpar][numpar], alpha_p[numpar][numpar], beta[numpar], deltas[numpar];

    chi2_0 = chi2_calc(x_0, I_obs, Q_obs, I_0, Q_0, std, w_I, w_Q);

    for (int itt = 0; itt < max_iter_inversion; itt++){
        fprintf(stdout, "\nitteration %i\t with a lambda of: %1.2e\n",itt,lambd);

        if(new_point){
            fprintf(stdout, "\nNew parameters:\n");
            fprintf(stdout, " a = %1.8e\n r = %1.8e\n eps = %1.8e\n delta = %1.8e\n Hd = %1.8e\n\n",\
            x_0[0], x_0[1], x_0[2], x_0[3], x_0[4]);
            surroundings_calc(x_0, xs, h);
            compute_alpha_beta(alpha, beta, I_obs, Q_obs, I_0, Q_0, x_0, xs, std, w_I, w_Q);
        }

        for (i = 0; i < numpar; i++){
            for (k = 0; k < numpar; k++){ alpha_p[i][j] = alpha[i][j];}
            alpha_p[i][i] = alpha_p[i][i]*(1 + lambd);
        }

        // SOLVE THE LINEAR SISTEM (alpha_P*deltas = beta) TO COMPUTE THE DELTAS

        for (i = 0; i < numpar; i++){
            if (x_0[i] + deltas[i] > x_u[i]){
                deltas[i] = x_u[i] - x_0[i];
            }else if (x_0[i] + deltas[i] < x_l[i]){
                deltas[i] = x_l[i] - x_0[i];
            }
        }

        for (i = 0; i < numpar; i++){
            x_1[i] = x_0[i] + deltas[i];
        }

        chi2_1 = chi2_calc(x_1, I_obs, Q_obs, I_1, Q_1, std, w_I, w_Q);
        
        if (chi2_1 >= chi2_0){
            lambd = lambd*10;
            new_point = 0;

            if (lambd > 1e25){
                break;
            }
        }else{
            lambd = lambd/10;
            for (i = 0; i < numpar; i++){
                x_0[i] = x_0[i] + deltas[i];
            }
            chi2_0 = chi2_1;
            for (j = 0; j < nw; j++){
                I_0[j] = I_1[j];
                Q_0[j] = Q_1[j];
            }
            new_point = 1;
        }
    }

    fprintf(stdout, "\n\n-----------------INVERSION FINISHED -------------------\n");
    return 0;
}
