/***************************************************************
*        2 LEVEL ATOM ATMOSPHERE SOLVER                        *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
****************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "integratives.h"
#include "params.h"
#include "subroutines.c"

/* -------------------------------------------------------------------*/
/* ------------------------- MAIN PROGRAM ----------------------------*/
/* -------------------------------------------------------------------*/
int main() {
    
    fprintf(stdout, "\n------------------- PARAMETERS OF THE PROBLEM ---------------------\n");
    fprintf(stdout, "optical thicknes of the lower boundary:            %1.1e \n", zl);
    fprintf(stdout, "optical thicknes of the upper boundary:            %1.1e \n", zu);
    fprintf(stdout, "resolution in the z axis:                          %1.3e \n", dz);
    fprintf(stdout, "total number of points in z:                       %i    \n", nz);
    fprintf(stdout, "lower/upper frequency limit :                      %1.3e   %1.3e \n", wl, wu);
    fprintf(stdout, "number points to sample the spectrum:              %i \n", nw);
    fprintf(stdout, "nodes in the gaussian quadrature (# dirs):         %i \n", qnd);
    /*fprintf(stdout, "T (isotermic) of the medium:                       %i \n", T);*/
    fprintf(stdout, "dumping Voigt profile:                             %1.3e \n", a);
    fprintf(stdout, "line strength XCI/XLI:                             %1.3e \n", r);
    fprintf(stdout, "Phot. dest. probability (LTE=1,NLTE=1e-4):         %1.3e \n", eps);
    fprintf(stdout, "Depolirarization colisions (delta):                %f \n", dep_col);
    fprintf(stdout, "Hanle depolarization factor [1/5, 1]:              %f \n", Hd);
    fprintf(stdout, "angular momentum of the levels (Ju, Jl):           (%i,%i) \n", ju, jl);
    fprintf(stdout, "Tolerance for finding the solution:                %f \n", tolerance);
    fprintf(stdout, "------------------------------------------------------------------\n\n");


    int i,j,k,l,m;               /* define the integers to count the loops*/

    double II[nz][nw][qnd], QQ[nz][nw][qnd];
    double S00[nz], S00_new[nz], S20[nz];
    double SLI[nz][qnd], SLQ[nz][qnd];
    double SI[nz][nw][qnd], SI_new[nz][nw][qnd], SQ[nz][nw][qnd];
    double lambda[nz][nw][qnd], lambda_w_integ[nz][nw], lambda_integ[nz];
    double J00[nz][nw], J20[nz][nw];
    double integrand_mu[qnd], integrand_w[nw];
    double Jm00[nz], Jm20[nz];

    double zz[nz], taus[nz][nw];
    double ww[nw], phy[nw], rr[nw], plank[nw];
    double mus[qnd], weigths[qnd];
    
    double psim[nw], psio[nw], psip[nw], w2jujl;
    double mrc, aux;

    /* compute the 1D grid in z*/
    for(i=0; i<nz; i++){
        zz[i] = zl + i*dz;
    }

    /* compute the grid in w and the profile of the line */
    for(j=0; j<nw; j++){
        ww[j] = wl + j*dw;
        plank[j] =  1;
        phy[j] = creal(voigt(ww[j],a));
    }

    /* normalice the profile for the integral to be 1 */
    double normalization = trapezoidal(phy, ww, nw);
    for (j = 0;j < nw; j++){
        phy[j] = phy[j]/normalization;
        rr[j] = phy[j]/(phy[j] + r);
        /*fprintf(stdout,"rr: %1.12e \n",rr[j]);*/
    }
    // fprintf(stdout, "Integral of the line profile:  %e \n", trapezoidal(phy, ww, nw));
    
    w2jujl = 1.;
    gauleg(-1, 1, mus, weigths, qnd);
    
    for (i=0; i<nz; i++){
        S00[i] = 1;
        S20[i] = 0;

        for (k = 0; k < qnd; k++){
            SLI[i][k] = S00[i] + w2jujl*(3*mus[k]*mus[k] -1)/(2*sqrt(2)) * S20[i];
            SLQ[i][k] = w2jujl*3*(mus[k]*mus[k] -1)/(2*sqrt(2)) * S20[i];

            for (j = 0; j < nw; j++){
                SI[i][j][k] = rr[j]*SLI[i][k] + (1-rr[j])*plank[j];
                SQ[i][j][k] = rr[j]*SLQ[i][k];
                
                if(i==0){ 
                    II[i][j][k] = plank[j];
                }else{
                    II[i][j][k] = 0;
                }
                QQ[i][j][k] = 0;

                taus[i][j] = exp(-zz[i])*(phy[j] + r);                
            }
        }
    }

    /* -------------------------------------------------------------------*/
    /* ---------------------------- MAIN LOOP ----------------------------*/
    /* -------------------------------------------------------------------*/
    for(l=1; l<=max_iter; l++){         /* loop with the total iterations */

        /*--------------------------- SOLVE THE RTE --------------------------*/
              
        RTE_SC_solve(II, QQ, SI, SQ, lambda, taus, mus);

        /* Check for negative intensities to stop and report a problem */
        for (i = 0; i < nz; i++){ 
            for (j = 0; j < nw; j++){
                for (k = 0; k < qnd; k++){
                    if( II[i][j][k] < -1e-4 ){
                        fprintf(stdout,"Found a negative intensity at: i: %i j: %i k: %i.",i,j,k); 
                        fprintf(stdout,"with value: %1.3e.  Stopping.\n", II[i][j][k]);
                        return 0;
                    }
                }                
            }
        }
        
        /* -------------------      COMPUTE THE J    -----------------------*/

        for (i = 0; i < nz; i++){
            for (j = 0; j < nw; j++){
                for (k = 0; k < qnd; k++){ integrand_mu[k] = (3.*mus[k]*mus[k] - 1)*II[i][j][k] +\
                                            3.*(mus[k]*mus[k] - 1)*QQ[i][j][k]; }
                
                J00[i][j] = num_gaus_quad( II[i][j], weigths, qnd);
                J20[i][j] = num_gaus_quad( integrand_mu, weigths, qnd);
                lambda_w_integ[i][j] = num_gaus_quad( lambda[i][j], weigths, qnd);
            }

            for (j = 0; j < nw; j++){ integrand_w[j] = phy[j]*J00[i][j];}
            Jm00[i] = 1./2. * trapezoidal(integrand_w, ww, nw);

            for (j = 0; j < nw; j++){ integrand_w[j] = phy[j]*J20[i][j];}
            Jm20[i] = 1./(4.*sqrt(2)) * trapezoidal(integrand_w, ww, nw);

            for (j = 0; j < nw; j++){ integrand_w[j] = rr[j]*phy[j]*lambda_w_integ[i][j];}
            lambda_integ[i] = 1./2. * trapezoidal(integrand_w, ww, nw);
        }
        /* -------------------     COMPUTE THE NEW S   -----------------------*/

        for (i = 0; i < nz; i++){

            S00_new[i] = (1-eps)*Jm00[i] + eps;
            S00_new[i] = (S00_new[i] - S00[i])/(1 - (1-eps)*lambda_integ[i]) + S00[i];
            S20[i] = Hd * (1-eps)/(1 + (1-eps)*dep_col) * w2jujl * Jm20[i];
            
            for (j = 0; j < nw; j++){

                for (k = 0; k < qnd; k++){

                    SLI[i][k] = S00_new[i] +  w2jujl * (3*mus[k]*mus[k] - 1)/sqrt(8.) * S20[i];
                    SLQ[i][k] = w2jujl * 3*(mus[k]*mus[k] - 1)/sqrt(8.) * S20[i];

                    SI_new[i][j][k] = rr[j]*SLI[i][k] + (1 - rr[j])*plank[j];
                    SQ[i][j][k] = rr[j]*SLQ[i][k];                
                }   
            }
        }
        /* -------------------------------------------------------------------*/
        /* ------------------- COMPUTE THE DIFFERENCES -----------------------*/
        /* -------------------------------------------------------------------*/
        mrc = -1;
        for (i = 0; i < nz; i++){
            for (j = 0; j < nw; j++){
                for (k = 0; k < qnd; k++){
                    aux = fabs((SI[i][j][k] - SI_new[i][j][k])/SI_new[i][j][k]);
                    if (aux > mrc){
                        mrc = aux;
                    }
                }
            }
        }
        
        printf("iteration: %i, Actual tolerance is :  %1.2e \n",l, mrc);
        if (mrc < tolerance){
            break;
        }
        for (i = 0; i < nz; i++){
            
            S00[i] = S00_new[i];

            for (j = 0; j < nw; j++){
                for (k = 0; k < qnd; k++){

                    SI[i][j][k] = SI_new[i][j][k];
                
                }   
            }   
        }        
    }

    fprintf(stdout,"\n----------------- FINISHED ------------------\n");

    return 0;
}