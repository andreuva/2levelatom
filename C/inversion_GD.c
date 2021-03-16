/***************************************************************
*        2 LEVEL ATOM INVERSIONS SOLVER                        *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
****************************************************************/
#include <stdio.h>
#include <time.h>
#include "forward_solver.c"
// implicits includes in forward_solver

// Define the number of free parameters of our inversion
#define numpar 5

// Define the selected mu to observe, the step of the
// numerical derivative, the weights of chi_2.
const int mu_sel = 9;
const double h = 1e-8  , std = 1e-5;
const double w_I = 1e-1, w_Q = 1e4;
const int max_iter_inversion = 1234;

// Define the solution parameters
const double a_sol = 1e-3;                      /* # dumping Voigt profile a=gam/(2^1/2*sig) */
const double r_sol = 1e-3;                     /* # line strength XCI/XLI */
const double eps_sol = 1e-4;                    /* # Phot. dest. probability (LTE=1,NLTE=1e-4) */
const double dep_col_sol = 0.8;                /* # Depolirarization colisions (delta) */
const double Hd_sol = 0.5;                      /* # Hanle depolarization factor [1/5, 1] */


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
                      double I_obs_sol[nw], double Q_obs_sol[nw], double std, double w_I, double w_Q){
    
    double I_obs[nw], Q_obs[nw];
    double chi2_pivot;
    double chi2s[numpar];
    int i;

    chi2_pivot = chi2_calc(params, I_obs_sol, Q_obs_sol, I_obs, Q_obs, std, w_I, w_Q);

    for (i=0; i<numpar; i++) {
        chi2s[i] = chi2_calc(surroundings[i], I_obs_sol, Q_obs_sol, I_obs, Q_obs, std, w_I, w_Q);
    }

    for (i=0; i<numpar; i++) {
        beta[i] = (chi2s[i] - chi2_pivot)/2*h;
    }

    return;
}

int main(){

    int i, j, k, itt;
    double I_full[nz][nw][qnd], Q_full[nz][nw][qnd];
    double I_0[nw], Q_0[nw], I_1[nw], Q_1[nw], I_obs_sol[nw], Q_obs_sol[nw];
    double x_0[numpar], x_1[numpar], x_sol[numpar] = {a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol};
    double x_l[numpar] = {1e-12, 1e-12, 1e-4, 0, 0.2};
    double x_u[numpar] = {1, 1, 1, 10 ,1};
    double step_size[numpar] = {1e-2,1e-7,1e-6,1e-1,1e-1};
    double xs[numpar][numpar], beta[numpar];
    float progres;

    double chi2_0, chi2_1, cc=1;
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

    add_noise_1D(nw, I_obs_sol, std);
    add_noise_1D(nw, Q_obs_sol, std);

    // initialice the parameters to start the inversion
    // for (i = 0; i < numpar; i++){
    //     x_0[i] = (double)rand()/(double)(RAND_MAX) * (x_u[i] - x_l[i]) + x_l[i];
    // }
    x_0[0] = 1e-5; x_0[1] = 1e-5; x_0[2] = 1e-3;  x_0[3] = 1e-1; x_0[4] = 0.6;

    fprintf(stdout, "\nINITIAL PARAMETERS:\n");
    fprintf(stdout, " a = %1.8e\n r = %1.8e\n eps = %1.8e\n delta = %1.8e\n Hd = %1.8e\n\n",\
                          x_0[0],     x_0[1],        x_0[2],        x_0[3],       x_0[4]);
    fprintf(stdout,"=====================================\n");
    
    chi2_0 = chi2_calc(x_0, I_obs_sol, Q_obs_sol, I_0, Q_0, std, w_I, w_Q);

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
            surroundings_calc(x_0, xs, h);
            check_range(x_0, xs, x_l, x_u);
            compute_gradient(x_0, xs, beta, I_obs_sol, Q_obs_sol, std, w_I, w_Q);
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
       
        chi2_1 = chi2_calc(x_1, I_obs_sol, Q_obs_sol, I_1, Q_1, std, w_I, w_Q);
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
    fp=fopen("/home/andreuva/Documents/2 level atom/2levelatom/figures/profiles.txt","w");
    
    fprintf(fp, "# index, I_solution, Q_solution, I_inverted, Q_inverted\n");
    for(i = 0; i < nw; i++){
       fprintf (fp, " %i , %1.8e , %1.8e , %1.8e , %1.8e\n",\
                    i, I_obs_sol[i], Q_obs_sol[i], I_0[i], Q_0[i]);
    }
    fclose (fp);

    fp=fopen("/home/andreuva/Documents/2 level atom/2levelatom/figures/parameters_inverted.txt","w");

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
                \n weigths:\t w_I = %f\t w_Q = %f\n",max_iter_inversion,std,mu_sel,h,w_I,w_Q);
    fprintf(fp, "------------------------------------------------------------------------\n");
    fclose (fp);

    return 0;
}
