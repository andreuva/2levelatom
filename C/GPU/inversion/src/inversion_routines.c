#include "includes_definitions.h"



// Function to generate random numbers (return a pointer to a array)
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

// Function to add the noise to an existing array
void add_noise_1D(int nn, double array[nn], double std){
    
    int i;
    double max=-1e50, min=1e50;
    double* normal_noise;

    // initialice the random number generator
    srand((unsigned int)time(NULL));
    normal_noise = generate(nn);

    for (i = 0; i < nn; i++){
        normal_noise[i] = normal_noise[i] * std;
        array[i] = array[i] + normal_noise[i];
    }
    free(normal_noise);
    return;
}



/* Function to compute the chi squared of a given parameters:
                        INPUTS
params   : parameters of the forward model to compare
I_obs_sol: Array of "observed" I varing with w
Q_obs_sol: Array of "observed" Q varing with w
std      : standar deviation of the noise in the observed profile 
w_I      : weight of the I component in the chi_2 calculation
w_Q      : weight of the Q component in the chi_2 calculation
                        OUTPUTS
I_obs : observed I profile with the given parameters (IMPLICIT)
Q_obs : observed Q profile with the given parameters (IMPLICIT)
Chi2  : Chi sqared of the given parameters (EXCLICIT)           */
double chi2_calc(double params[numpar], double I_obs_sol[nw], double Q_obs_sol[nw],
double I_obs[nw], double Q_obs[nw], double std, double w_I, double w_Q){

    double II[nz][nw][qnd], QQ[nz][nw][qnd];
    double chi2 = 0.0;
    int j;

    solve_profiles(params[0], params[1], params[2], params[3], params[4], II, QQ);
    for (j = 0; j < nw; j++){
        I_obs[j] = II[nz-1][j][mu_sel];
        Q_obs[j] = QQ[nz-1][j][mu_sel];
    }
    
    for (j = 0; j < nw; j++)
    {
        chi2 = chi2 + ( w_I*(I_obs[j] - I_obs_sol[j])*(I_obs[j] - I_obs_sol[j]) \
        + w_Q*(Q_obs[j]-Q_obs_sol[j])*(Q_obs[j]-Q_obs_sol[j]) )/(std*std*2*nw);
    }
    // fprintf(stdout,"Chi^2 of this profiles is: %1.5e\n", chi2 );
    return chi2;
}

// Compute the surroundings of a point in each parameter dimension to obtain the derivatives
// returns a matrix with all the surrounding points as x_j[i] += x_j[i]+h[i] 
void surroundings_calc(double x_0[numpar], double surroundings[numpar][numpar], double hh){

    double delta[numpar];
    int i,j;
    
    for(i = 0; i < numpar; i++){
        for(j = 0; j < numpar; j++){ delta[j] = 0; }
        delta[i] = hh;
        for(j = 0; j < numpar; j++){ surroundings[i][j] = x_0[j] + delta[j]; }
    }
    return;
}

// Check if the surroundings are inside of the valid range of the parameters (X_l,x_u)
// If not adjust the initial point (X_0) and the surroundings to fit inside the range
// and to have the same direction of the derivative
void check_range(double x_0[], double xs[][numpar], double x_l[], double x_u[]){
    int i,j,k;

    for (i = 0; i < numpar; i++){
        if (xs[i][i] > x_u[i]){
            x_0[i] = x_u[i] - (xs[i][i] - x_0[i]);
            xs[i][i] = x_u[i];
        }else if( xs[i][i] < x_l[i] ){
            x_0[i] = x_l[i] + (x_0[i] - xs[i][i]);
            xs[i][i] = x_l[i];
        }
    }
    return;
}

void compute_gradient(double params[numpar], double surroundings[numpar][numpar], double beta[numpar],
                      double I_obs_sol[nw], double Q_obs_sol[nw], double hh, double std, double w_I, double w_Q){
    
    double I_obs[nw], Q_obs[nw];
    double chi2_pivot;
    double chi2s[numpar];
    int i;

    chi2_pivot = chi2_calc(params, I_obs_sol, Q_obs_sol, I_obs, Q_obs, std, w_I, w_Q);

    for (i=0; i<numpar; i++) {
        chi2s[i] = chi2_calc(surroundings[i], I_obs_sol, Q_obs_sol, I_obs, Q_obs, std, w_I, w_Q);
    }

    for (i=0; i<numpar; i++) {
        beta[i] = (chi2s[i] - chi2_pivot)/2*hh;
    }

    return;
}