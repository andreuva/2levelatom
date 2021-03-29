/**********************************************************************/
/*  Program to solve the inversion of the 2 level atom problem        */
/*  Author: Andres Vicente Arevalo      Date: 19-03-2021              */
/**********************************************************************/
#include "includes_definitions.h"

// Define the "observed" parameters
const double a_sol = 1e-3;                      /* # dumping Voigt profile a=gam/(2^1/2*sig) */
const double r_sol = 1e-3;                     /* # line strength XCI/XLI */
const double eps_sol = 1e-4;                    /* # Phot. dest. probability (LTE=1,NLTE=1e-4) */
const double dep_col_sol = 0.8;                /* # Depolirarization colisions (delta) */
const double Hd_sol = 0.5;                      /* # Hanle depolarization factor [1/5, 1] */

int main(){

    int i, j, k, itt;
    double I_full[nz][nw][qnd], Q_full[nz][nw][qnd];
    double I_0[nw], Q_0[nw], I_1[nw], Q_1[nw];
    double I_obs_sol[nw], Q_obs_sol[nw], I_init[nw], Q_init[nw];
    double Jm00_0[NODES], Jm20_0[NODES], Jm00_1[NODES], Jm20_1[NODES];
    double x_0[numpar], x_1[numpar];
    double x_sol[numpar] = {a_sol, r_sol, eps_sol, dep_col_sol, Hd_sol};
    double x_l[numpar] = {1e-12, 1e-12, 1e-4, 0, 0.2};
    double x_u[numpar] = {1, 1, 1, 10 ,1};
    double step_size[numpar] = {1e-2,1e-7,1e-6,1e-3,1e-3};
    double Jm_step_size = 1e-4;
    double xs[numpar][numpar], beta[numpar];
    double Jm00s[NODES][NODES], betaj00[NODES];
    double Jm20s[NODES][NODES], betaj20[NODES];
    float progres;

    double chi2_0, chi2_1, cc=1;
    int new_point = 1; 

    fprintf(stdout,"=====================================\n");
    fprintf(stdout, "\nSOLUTION PARAMETERS:\n");
    fprintf(stdout, " a = %1.8e\n r = %1.8e\n eps = %1.8e\n delta = %1.8e\n Hd = %1.8e\n\n",\
                          x_sol[0],    x_sol[1],     x_sol[2],      x_sol[3],    x_sol[4]);
    fprintf(stdout,"=====================================\n");

    fprintf(stdout, "Computing the solution profiles... \n\n");
    solve_profiles_sol( x_sol[0], x_sol[1], x_sol[2], x_sol[3], x_sol[4], I_full, Q_full);
    for (j = 0; j < nw; j++){
        I_obs_sol[j] = I_full[nz-1][j][mu_sel];
        Q_obs_sol[j] = Q_full[nz-1][j][mu_sel];
    }
    fprintf(stdout,"=====================================\n");

    add_noise_1D(nw, I_obs_sol, STD);
    add_noise_1D(nw, Q_obs_sol, STD);

    // initialice the parameters to start the inversion
    x_0[0] = 1e-5; x_0[1] = 1e-5; x_0[2] = 1e-3;  x_0[3] = 1e-1; x_0[4] = 0.6;
    for (i = 0; i < NODES; i++) {
        Jm00_0[i] = 1;
        Jm20_0[i] = 0;
    }

    fprintf(stdout, "\nINITIAL PARAMETERS:\n");
    fprintf(stdout, " a = %1.8e\n r = %1.8e\n eps = %1.8e\n delta = %1.8e\n Hd = %1.8e\n\n",\
                          x_0[0],     x_0[1],        x_0[2],        x_0[3],       x_0[4]);
    fprintf(stdout,"=====================================\n");
    
    chi2_0 = chi2_calc(x_0, Jm00_0, Jm20_0, I_obs_sol, Q_obs_sol, 
                       I_0, Q_0, WJ00, WJ20, STD, WI, WQ, 1);

    for (i = 0; i < nw; i++) {
        I_init[i] = I_0[i];
        Q_init[i] = Q_0[i];
    }

    for (itt = 0; itt < max_iter_inversion; itt++){

        progres_bar(itt, max_iter_inversion);

        if(new_point == 1){
            surroundings_calc(numpar, x_0, xs, HH);
            surroundings_calc(NODES, Jm00_0, Jm00s, HH);
            surroundings_calc(NODES, Jm20_0, Jm20s, HH);
            check_range(x_0, xs, x_l, x_u);
            compute_gradient(x_0, xs, beta,
                             Jm00_0, Jm00s, betaj00,
                             Jm20_0, Jm20s, betaj20,
                             I_obs_sol, Q_obs_sol,
                             HH, STD, WJ00, WJ20, WI, WQ);
        }

        for (i = 0; i < numpar; i++){
            x_1[i] = x_0[i] - step_size[i]*beta[i]*cc;
        }

        for (i = 0; i < NODES; i++) {
            Jm00_1[i] = Jm00_0[i] - Jm_step_size*betaj00[i]*cc;
            Jm20_1[i] = Jm20_0[i] - Jm_step_size*betaj20[i]*cc;
        }

        for (i=0; i<numpar; i++){
            if (x_1[i] > x_u[i]){
                x_1[i] = x_u[i];
            }else if (x_1[i] < x_l[i]){
                x_1[i] = x_l[i];
            }
        }

        chi2_1 = chi2_calc(x_1, Jm00_1, Jm20_1, I_obs_sol, Q_obs_sol, 
                           I_1,  Q_1, WJ00, WJ20, STD, WI, WQ, 0);

        fprintf(stdout,"Total Chi^2 %1.6e\n with step parameter of %1.3e\n",chi2_1,cc);

        // If we have very good loss stop
        if (chi2_1 < 10){
            break;
        }

        if (chi2_1 <= chi2_0 || itt == 0){
            for (i = 0; i < numpar; i++){
                x_0[i] = x_1[i];
            }
            for (i = 0; i < NODES; i++) {
                Jm00_0[i] = Jm00_1[i];
                Jm20_0[i] = Jm20_1[i];
            }
            for (j = 0; j < nw; j++){
                I_0[j] = I_1[j];
                Q_0[j] = Q_1[j];
            }
            cc = cc*1.25;
            new_point = 1;
            chi2_0 = chi2_1;

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
    fprintf(stdout, "  a = %1.10e \t\t |  a_sol = %1.10e\n  r = %1.10e \t\t |  r_sol = %1.10e\
    \n  eps = %1.10e \t\t |  eps_sol = %1.10e\n  delta = %1.10e \t |  delta_sol = %1.10e\
    \n  Hd = %1.10e \t\t |  Hd_sol = %1.10e\n",\
    x_0[0], x_sol[0], x_0[1], x_sol[1], x_0[2], x_sol[2], x_0[3], x_sol[3], x_0[4], x_sol[4]);
    fprintf(stdout, "--------------------------------------------------------------------\n");
    
    /* define array, counter, and file name, and open the file */

    FILE *fp;

    fp=fopen("./figures/profiles_jkq.txt","w");
    fprintf(fp, "# index, I_solution, Q_solution, I_inverted, Q_inverted, I_initial, Q_initial\n");
    for(i = 0; i < nw; i++){
       fprintf (fp, " %i , %1.10e , %1.10e , %1.10e , %1.10e , %1.10e , %1.10e\n",\
                    i, I_obs_sol[i], Q_obs_sol[i], I_0[i], Q_0[i], I_init[i], Q_init[i]);
    }
    fclose (fp);

    fp=fopen("./figures/radiation_jkq.txt","w");
    fprintf(fp, "# index, zz,  Jm00, Jm20\n");
    for(i = 0; i < nw; i++){
       fprintf (fp, " %i , %1.3e , %1.10e , %1.10e\n",\
                      i, zl + dz*i, Jm00_0[i], Jm20_0[i]);
    }
    fclose (fp);

    fp=fopen("./figures/parameters_jkq_inverted.txt","w");
    fprintf(fp, "\nFINISHED AFTER %i ITTERATIONS\n", itt);
    fprintf(fp, "\n------------------------------------------------------------------------\n");
    fprintf(fp, "\tINVERTED PARAMETERS\t | \tSOLUTION PARAMETERS\t\n");
    fprintf(fp, "------------------------------------------------------------------------\n");
    fprintf(fp, "  a = %1.10e \t\t\t |  a_sol = %1.10e\n  r = %1.10e \t\t\t |  r_sol = %1.10e\
    \n  eps = %1.10e \t\t\t |  eps_sol = %1.10e\n  delta = %1.10e \t\t |  delta_sol = %1.10e\
    \n  Hd = %1.10e \t\t\t |  Hd_sol = %1.10e\n",\
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
