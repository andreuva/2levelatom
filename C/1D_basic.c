/***************************************************************
*        2 LEVEL ATOM ATMOSPHERE SOLVER                        *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>    /* Standard Library of Complex Numbers */

/*  DEFINE SOME OF THE CONSTANTS OF THE PROBLEM  */
const int c = 299792458;                   /* # m/s */
const float h = 6.626070150e-34;             /* # J/s */
const float kb = 1.380649e-23;                /* # J/K */
const float R = 8.31446261815324;            /* # J/K/mol */

/*  DEFINE THE PROBLEM PARAMETERS                */
const float zl = -15; /*-log(1e3);                    /* optical thicknes of the lower boundary */
const float zu = 8; /*-log(1e-3);                   /* optical thicknes of the upper boundary */

const float eps = 1e-4;                  /* # Phot. dest. probability (LTE=1,NLTE=1e-4) */

const float tolerance = 1e-6;           /* # Tolerance for finding the solution */
const int max_iter = 5000;              /* maximum number of iterations */

const float dz = .1 /*(zu-zl)/(nz-1)*/;    /* resolution in the z axis*/
const short nz = (zu-zl)/dz;                      /* # number of points in the z axes */

void psicalc( double *psim, double *psio, double *psip, double deltaum, double deltaup, int mode){
    
    double U0 = 1 - exp(-deltaum);
    double U1 = deltaum - U0;

    if (mode == 1){
        *psim = U0 - U1/deltaum;
        *psio = U1/deltaum;
    }else{

        double U2 = deltaum*deltaum - 2.*U1;

        *psim = U0 + (U2 - U1*(deltaup + 2.*deltaum))/(deltaum*(deltaum + deltaup));
        *psio = (U1*(deltaum + deltaup) - U2)/(deltaum*deltaup);
        *psip = (U2 - U1*deltaum)/(deltaup*(deltaup+deltaum));
    }
}

/* -------------------------------------------------------------------*/
/* ------------------------- MAIN PROGRAM ----------------------------*/
/* -------------------------------------------------------------------*/
int main() {

    fprintf(stdout, "\n------------------- PARAMETERS OF THE PROBLEM ---------------------\n");
    fprintf(stdout, "optical thicknes of the lower boundary:            %1.1e \n", zl);
    fprintf(stdout, "optical thicknes of the upper boundary:            %1.1e \n", zu);
    fprintf(stdout, "resolution in the z axis:                          %1.3e \n", dz);
    fprintf(stdout, "Phot. dest. probability (LTE=1,NLTE=1e-4):         %f \n", eps);
    fprintf(stdout, "Tolerance for finding the solution:                %f º/. \n", tolerance*100);
    fprintf(stdout, "------------------------------------------------------------------\n\n");

    int i,itt;

    double I_up[nz], I_down[nz], J[nz];  
    double lambda_up[nz], lambda_down[nz], lambda_integ[nz];
    double SI[nz], SI_new[nz], SI_analitic[nz];

    double zz[nz], tau[nz];
    
    double psim,psio,psip, psip_prev, deltaum, deltaup;
    double change, mrc;
    double mu_up = 1/sqrt(3), mu_down = -1/sqrt(3);

    /* Initialice the variables and put boundary conditions*/
    for(i=0; i<nz; i++){
        zz[i] = zl + i*dz;
        tau[i] = exp(-zz[i]);
        SI[i] = 1;                      
        SI_new[i] = 1;
        lambda_down[i] = 0;             
        lambda_up[i] = 0;               
        lambda_integ[i] = 0;
        I_up[i] = 0;                    
        I_down[i] = 0;
        SI_analitic[i] = (1-eps)*(1-exp(-tau[i]*sqrt(3*eps))/(1+sqrt(eps))) + eps;
    }
    I_up[0] = 1;
    /* -------------------------------------------------------------------*/
    /* ---------------------------- MAIN LOOP ----------------------------*/
    /* -------------------------------------------------------------------*/
    for(itt=1; itt<=max_iter; itt++){         /* loop with the total iterations */
        
        /* -------------------------------------------------------------------*/
        /*--------------------------- SOLVE THE RTE --------------------------*/
        /* -------------------------------------------------------------------*/ 
        /*------------------------- SOLVE FOR THE UPWARD RAYS ----------------*/
        psip_prev = 0;  
        for (i = 1; i < nz; i++){
            deltaum = (tau[i-1] - tau[i])/mu_up;
            
            if (i < nz-1){
                deltaup = (tau[i] - tau[i+1])/mu_up;
                psicalc(&psim, &psio, &psip, deltaum, deltaup, 2);
                
                I_up[i] = I_up[i-1]*exp(-deltaum) + psim*SI[i-1] + psio*SI[i] + psip*SI[i+1];
                lambda_up[i] = psip_prev*exp(-deltaum) + psio;

                psip_prev = psip;
            }else{
                psicalc(&psim, &psio, &psip, deltaum, deltaup, 1);

                I_up[i] = I_up[i-1]*exp(-deltaum) + psim*SI[i-1] + psio*SI[i];
                lambda_up[i] = psip_prev*exp(-deltaum) + psio;
            }   
        }
        /*-------------------- REPEAT FOR THE DOWNWARD RAYS ----------------*/
        psip_prev = 0;  
        for (i = nz-2; i >= 0; i--){
            deltaum = -(tau[i] - tau[i+1])/mu_down;
            
            if (i > 0){
                deltaup = -(tau[i-1] - tau[i])/mu_down;
                psicalc(&psim, &psio, &psip, deltaum, deltaup, 2);
                
                I_down[i] = I_down[i+1]*exp(-deltaum) + psim*SI[i+1] + psio*SI[i] + psip*SI[i-1];
                lambda_down[i] = psip_prev*exp(-deltaum) + psio;
                psip_prev = psip;

            }else{
                psicalc(&psim, &psio, &psip, deltaum, deltaup, 1);
                
                I_down[i] = I_down[i+1]*exp(-deltaum) + psim*SI[i+1] + psio*SI[i];
                lambda_down[i] = psip_prev*exp(-deltaum) + psio;
            }   
        }

        /* -------------------------------------------------------------------*/
        /* -------------- COMPUTE THE J AND THE NEW SI -----------------------*/
        /* -------------------------------------------------------------------*/
        for (i = 0; i < nz; i++){
            J[i] = 1./2. * (I_up[i] + I_down[i]);
            lambda_integ[i] = 1./2. * (lambda_down[i] + lambda_up[i]);
            SI_new[i] = (1-eps)*J[i] + eps;
            SI_new[i] = (SI_new[i] - SI[i])/(1 - (1-eps)*lambda_integ[i]) + SI[i];
        }

        /* -------------------------------------------------------------------*/
        /* ------------------- COMPUTE THE DIFFERENCES -----------------------*/
        /* -------------------------------------------------------------------*/
        mrc=-1;
        for (i = 0; i < nz; i++){
            change = (SI[i] - SI_new[i])/SI[i];            
            if (change > mrc){
                mrc = change;
            }
            SI[i] = SI_new[i];
        }
        /*fprintf(stdout,"ACTUAL MRC: %1.2e \n", mrc);*/
        if ( mrc < tolerance ){
        break;
        }
    }
    if ( mrc < tolerance ){
        fprintf(stdout,"CONVERGENCE OBTAINED IN %d ITERATIONS\
         WITH AN ERROR OF %1.3e \n", itt, mrc);
    }
    else{
        fprintf(stdout,"CONVERGENCE NOT OBTAINED TO THE DESIRE DEGREE \
        AFTER %d ITERATIONS. ERROR OF %1.3e \n", itt, mrc);
    }

    return 0;
}