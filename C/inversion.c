/***************************************************************
*        2 LEVEL ATOM ATMOSPHERE SOLVER                        *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
****************************************************************/
#include <time.h>
#include "forward_solver.c"
#include "linalg.c"
// implicits includes in forward_solver

// Define the number of free parameters of our inversion
#define numpar 5

// Define the selected mu to observe, the step of the
// numerical derivative, the weights of chi_2.
const int mu_sel = 9;
const double h = 1e-10;
const double w_I = 1e-1, w_Q = 1e2;
const int max_iter_inversion = 1000;


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
Chi2  : Chi sqared of the given parameters (EXCLICIT)              */
double chi2_calc(double params[numpar], double I_obs_sol[nw], double Q_obs_sol[nw],
double I_obs[nw], double Q_obs[nw], double std, double w_I, double w_Q){

    double II[nz][nw][qnd], QQ[nz][nw][qnd];
    double chi2 = 0.;
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
    fprintf(stdout,"Chi^2 of this profiles is: %1.5e\n", chi2 );

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

void compute_alpha_beta(double alpha[numpar][numpar], double beta[numpar], double I_sol[nw], double Q_sol[nw],\
double I_0[nw],double Q_0[nw], double x_0[numpar], double xs[numpar][numpar], double std, double w_I,double w_Q){

    double I_p[nz][nw][qnd], Q_p[nz][nw][qnd], II[nw], QQ[nw], Is[numpar][nw], Qs[numpar][nw];
    double dQ[numpar][nw], dI[numpar][nw];
    double sum;
    int i, j, k, l;

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
        sum = 0;
        for (l = 0; l < nw; l++){
            sum = sum + (w_I*(Is[i][l] - I_sol[l])*dI[i][l] + w_Q*(Qs[i][l] - Q_sol[l])*dQ[i][l])/(nw*std*std);
        }
        beta[i] = sum;
       
        for (j = 0; j < numpar; j++){
            sum = 0;
            for (l = 0; l < nw; l++){
                sum = sum + (w_I*dI[i][l]*dI[j][l] + w_Q*dQ[i][l]*dQ[j][l])/nw;
            }
            alpha[i][j] = sum;
        }
    }
    
    return;
}


void solve_linear_sistem(double aa[numpar][numpar], double bb[numpar], double res[numpar]) {
    int i, j, n = numpar, * indx;
    double d, * b, ** a;

    a = dmatrix(1, n, 1, n);
    b = dvector(1, n);
    
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            a[i+1][j+1] = aa[i][j];
        }
        b[i+1] = bb[i];
    }
    
    indx = ivector(1, n);
    ludcmp(a, n, indx, & d);
    lubksb(a, n, indx, b);

    for (i = 0; i < n; i++){
        res[i] = b[i+1];
    }

    free_dvector(b, 1, n);
    free_ivector(indx, 1, n);
    free_dmatrix(a, 1, n, 1, n);

    return;
}


int main(){

    double lambd = 1e-5;
    double x_l[numpar] = {1e-12, 1e-12, 1e-4, 0, 0.2};
    double x_u[numpar] = {1, 1, 1, 1 ,1};
    double std = 1e-5;

    int i, j, k;
    double *profile_solution;
    double I_full[nz][nw][qnd], Q_full[nz][nw][qnd];
    double I_0[nw], Q_0[nw], I_1[nw], Q_1[nw], I_obs_sol[nw], Q_obs_sol[nw];
    double x_0[numpar], x_1[numpar], x_sol[numpar] = {a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol};
    double xs[numpar][numpar];

    double chi2_0, chi2_1;
    int new_point = 1;
    double alpha[numpar][numpar], alpha_p[numpar][numpar], beta[numpar], deltas[numpar];
    
    for (i = 0; i < numpar; i++){
        beta[i] = 0;
        deltas[i] = 0;
        for (j = 0; j < numpar; j++){
            alpha[i][j] = 0;
            alpha_p[i][j] = 0;
        }
    }
 

    fprintf(stdout, "\nSOLUTION PARAMETERS:\n");
    fprintf(stdout, " a = %1.8e\n r = %1.8e\n eps = %1.8e\n delta = %1.8e\n Hd = %1.8e\n\n",\
    x_sol[0], x_sol[1], x_sol[2], x_sol[3], x_sol[4]);
    fprintf(stdout, "Computing the solution profiles:");
    solve_profiles( x_sol[0], x_sol[1], x_sol[2], x_sol[3], x_sol[4], I_full, Q_full);
    // add_noise(I_sol, std_I);
    // add_noise(Q_sol, std_Q);

    for (j = 0; j < nw; j++){
        I_obs_sol[j] = I_full[nz-1][j][mu_sel];
        Q_obs_sol[j] = Q_full[nz-1][j][mu_sel];
    }

    // add_noise_1D(nw, I_obs_sol, std, -1.);
    // add_noise_1D(nw, Q_obs_sol, std, -1.);

    // initialice the parameters to start the inversion
    // for (i = 0; i < numpar; i++){
    //     x_0[i] = (double)rand()/(double)(RAND_MAX) * (x_u[i] - x_l[i]) + x_l[i];
    // }
    fprintf(stdout, "\nINITIAL PARAMETERS:\n");
    x_0[0] = a_sol + a_sol*0.02 ; x_0[1] = r_sol - r_sol * 0.05; x_0[2] = eps_sol + eps_sol*0.01; 
    x_0[3] = dep_col_sol; x_0[4] = Hd_sol - Hd_sol*0.06 ;
    fprintf(stdout, " a = %1.8e\n r = %1.8e\n eps = %1.8e\n delta = %1.8e\n Hd = %1.8e\n\n",\
    x_0[0], x_0[1], x_0[2], x_0[3], x_0[4]);

    chi2_0 = chi2_calc(x_0, I_obs_sol, Q_obs_sol, I_0, Q_0, std, w_I, w_Q);

    for (int itt = 0; itt < max_iter_inversion; itt++){
        fprintf(stdout, "\nitteration %i\t with a lambda of: %1.2e\n",itt,lambd);

        if(new_point){
            fprintf(stdout, "\nNew parameters:\n");
            fprintf(stdout, " a = %1.8e\n r = %1.8e\n eps = %1.8e\n delta = %1.8e\n Hd = %1.8e\n\n",\
            x_0[0], x_0[1], x_0[2], x_0[3], x_0[4]);
            surroundings_calc(x_0, xs, h);
            check_range(x_0, xs, x_l, x_u);
            compute_alpha_beta(alpha, beta, I_obs_sol, Q_obs_sol, I_0, Q_0, x_0, xs, std, w_I, w_Q);
        }

        for (i = 0; i < numpar; i++){
            for (j = 0; j < numpar; j++){ alpha_p[i][j] = alpha[i][j];}
            alpha_p[i][i] = alpha_p[i][i]*(1 + lambd);
        }

        // for (i = 0; i < numpar; i++) {
        //     for (j = 0; j < numpar; j++){ printf("%1.3e\t", alpha_p[i][j]); }
        //     printf(" %5s  %1.3e\n", (i == 2 ? "* X =" : "     "), beta[i]);
        // }
        // printf( "\n");

        // SOLVE THE LINEAR SISTEM (alpha_P*deltas = beta) TO COMPUTE THE DELTAS
        solve_linear_sistem(alpha_p , beta, deltas);

        // for (i = 0; i < numpar; i++) {
        //     for (j = 0; j < numpar; j++){ printf("%1.3e\t", alpha_p[i][j]); }
        //     printf(" %5s  %1.3e\n", (i == 2 ? "* X =" : "     "), deltas[i]);
        // }
        // printf( "\n\n");

        for (i = 0; i < numpar; i++){
            deltas[i] = deltas[i]/1e12;
            x_1[i] = x_0[i] + deltas[i];
        }

        for (i = 0; i < numpar; i++){
            if (x_1[i] > x_u[i]){
                x_1[i] = x_u[i];
            }else if (x_1[i] < x_l[i]){
                x_1[i] = x_l[i];
            }
        }

        chi2_1 = chi2_calc(x_1, I_obs_sol, Q_obs_sol, I_1, Q_1, std, w_I, w_Q);
        
        if (chi2_1 >= chi2_0){
            lambd = lambd*10;
            new_point = 0;

            if (lambd > 1e35){
                break;
            }
        }else{
            lambd = lambd/10;
            for (i = 0; i < numpar; i++){
                x_0[i] = x_1[i];
            }
            chi2_0 = chi2_1;
            for (j = 0; j < nw; j++){
                I_0[j] = I_1[j];
                Q_0[j] = Q_1[j];
            }
            new_point = 1;
        }
        // return 0;
    }

    fprintf(stdout, "\n\n-------------------------INVERSION FINISHED-------------------------\n");
    fprintf(stdout, "\tINVERTED PARAMETERS\t | \tSOLUTION PARAMETERS\t\n");
    fprintf(stdout, "--------------------------------------------------------------------\n");
    fprintf(stdout, "  a = %1.8e \t\t |  a_sol = %1.8e\n  r = %1.8e \t\t |  r_sol = %1.8e\
    \n  eps = %1.8e \t\t |  eps_sol = %1.8e\n  delta = %1.8e \t |  delta_sol = %1.8e\
    \n  Hd = %1.8e \t\t |  Hd_sol = %1.8e\n",\
    x_0[0], x_sol[0], x_0[1], x_sol[1], x_0[2], x_sol[2], x_0[3], x_sol[3], x_0[4], x_sol[4]);
    fprintf(stdout, "--------------------------------------------------------------------\n");
    return 0;
}
