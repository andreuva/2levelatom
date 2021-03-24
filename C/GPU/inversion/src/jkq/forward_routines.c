/**********************************************************************/
/*  Subroutines to solve the forward modeling for the SC method       */
/*  Author: Andres Vicente Arevalo      Date: 20-11-2020              */
/**********************************************************************/
#include "includes_definitions.h"

/* Function to compute the voigt profile of a line giveng the dumping 
 * nd the frequency at wich to compute it                             */
double complex voigt(double v, double a){

    double s = fabs(v)+a;
    double d = .195e0*fabs(v)-.176e0;
    double complex t, z = a - v*I;

    if (s >= .15e2){
        t = .5641896e0*z/(.5+z*z);
    }
    else{

        if (s >= .55e1){

            double complex u = z*z;
            t = z*(.1410474e1 + .5641896e0*u)/(.75e0 + u*(.3e1 + u));
        }
        else{

            if (a >= d){
                double complex nt = .164955e2 + z*(.2020933e2 + z*(.1196482e2 + \
                                    z*(.3778987e1 + .5642236e0*z)));
                double complex dt = .164955e2 + z*(.3882363e2 + z*(.3927121e2 + \
                                    z*(.2169274e2 + z*(.6699398e1 + z))));
                t = nt / dt;
            }
            else{
                double complex u = z*z;
                double complex x = z*(.3618331e5 - u*(.33219905e4 - u*(.1540787e4 - \
                    u*(.2190313e3 - u*(.3576683e2 - u*(.1320522e1 - \
                    .56419e0*u))))));
                double complex y = .320666e5 - u*(.2432284e5 - u*(.9022228e4 - \
                    u*(.2186181e4 - u*(.3642191e3 - u*(.6157037e2 - \
                    u*(.1841439e1 - u))))));
                t = cexp(u) - x/y;
            }
        }
    }
    return t;
}

/* Function to compute the coeficients of the SC method for lineal and quadratic */
void psi_calc(double deltaum[], double deltaup[], \
              double psim[], double psio[], double psip[], int mode)
{
    // Compute of the psi coefficients in the SC method

    double U0[nw], U1[nw], U2[nw];
    int j;

    for (j = 0; j < nw; j++){
        
        if (deltaum[j] < 1e-3){
            U0[j] = deltaum[j] - deltaum[j]*deltaum[j]/2 +\
                    deltaum[j]*deltaum[j]*deltaum[j]/6;
        }
        else{
            U0[j] = 1 - exp(-deltaum[j]);
        }
        U1[j] = deltaum[j] - U0[j];
    }
     
    
    if (mode == 1){
        for (j = 0; j < nw; j++){
            psim[j] = U0[j] - U1[j]/deltaum[j];
            psio[j] = U1[j]/deltaum[j];
            psip[j] = 0;
        }
    }
    else if (mode == 2){
        for (j = 0; j < nw; j++){
            U2[j] = deltaum[j]*deltaum[j] - 2*U1[j];

            psim[j] = U0[j] + (U2[j] - U1[j]*(deltaup[j] + 2*deltaum[j]))/\
                                        (deltaum[j]*(deltaum[j] + deltaup[j]));
            psio[j] = (U1[j]*(deltaum[j] + deltaup[j]) - U2[j])/(deltaum[j]*deltaup[j]);
            psip[j] = (U2[j] - U1[j]*deltaum[j])/(deltaup[j]*(deltaup[j]+deltaum[j]));
        }        
    }
    else{
        fprintf(stdout,"ERROR IN MODE OF THE PSICALC FUNCTION");
    }

    return;
}

/* Solver of the Radiative transfer equations by the SC method and compute of the lambda operator */
void RTE_SC_solve(double II[][nw][qnd], double QQ[][nw][qnd], double SI[nz][nw][qnd],\
                 double SQ[nz][nw][qnd], double tau[nz][nw], double mu[qnd]){
    
    // Compute the new intensities form the source function with the SC method
    
    double psim[nw], psio[nw], psip[nw];
    double deltaum[nw], deltaup[nw];
    int i,j,k;

    for (k = 0; k < qnd; k++){
        if (mu[k] > 0){

            for (i = 1; i < nz; i++){
                for (j = 0; j < nw; j++){ deltaum[j] = fabs((tau[i-1][j]-tau[i][j])/mu[k]); }
                
                if (i < nz-1){
                    for (j = 0; j < nw; j++){ deltaup[j] = fabs((tau[i][j]-tau[i+1][j])/mu[k]); }
                    psi_calc(deltaum, deltaup, psim, psio, psip, 2);

                    for (j = 0; j < nw; j++){
                        II[i][j][k] = II[i-1][j][k]*exp(-deltaum[j]) + SI[i-1][j][k]*psim[j] + SI[i][j][k]*psio[j] + SI[i+1][j][k]*psip[j];
                        QQ[i][j][k] = QQ[i-1][j][k]*exp(-deltaum[j]) + SQ[i-1][j][k]*psim[j] + SQ[i][j][k]*psio[j] + SQ[i+1][j][k]*psip[j];
                    }
                    
                }
                else{
                    psi_calc(deltaum, deltaup, psim, psio, psip, 1);
                    for (j = 0; j < nw; j++){
                        II[i][j][k] = II[i-1][j][k]*exp(-deltaum[j]) + SI[i-1][j][k]*psim[j] + SI[i][j][k]*psio[j];
                        QQ[i][j][k] = QQ[i-1][j][k]*exp(-deltaum[j]) + SQ[i-1][j][k]*psim[j] + SQ[i][j][k]*psio[j];
                    }
                }
                             
            }
        }
        else{

            for (i = nz-2; i >= 0; i--){
                
                for (j = 0; j < nw; j++){ deltaum[j] = fabs((tau[i][j]-tau[i+1][j])/mu[k]); }
                
                if (i > 0){

                    for (j = 0; j < nw; j++){ deltaup[j] = fabs((tau[i-1][j]-tau[i][j])/mu[k]); }
                    psi_calc(deltaum, deltaup, psim, psio, psip, 2);

                    for (j = 0; j < nw; j++){
                        II[i][j][k] = II[i+1][j][k]*exp(-deltaum[j]) + SI[i+1][j][k]*psim[j] + SI[i][j][k]*psio[j] + SI[i-1][j][k]*psip[j];
                        QQ[i][j][k] = QQ[i+1][j][k]*exp(-deltaum[j]) + SQ[i+1][j][k]*psim[j] + SQ[i][j][k]*psio[j] + SQ[i-1][j][k]*psip[j];
                    }
                    
                }
                else{
                    psi_calc(deltaum, deltaup, psim, psio, psip, 1);

                    for (j = 0; j < nw; j++){
                        II[i][j][k] = II[i+1][j][k]*exp(-deltaum[j]) + SI[i+1][j][k]*psim[j] + SI[i][j][k]*psio[j];
                        QQ[i][j][k] = QQ[i+1][j][k]*exp(-deltaum[j]) + SQ[i+1][j][k]*psim[j] + SQ[i][j][k]*psio[j];
                    }
                }            
            }
        }
    }
    return;
}
