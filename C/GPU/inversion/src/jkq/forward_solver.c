/* -------------------------------------------------------------------*/
/* --------------- SUBROUTINE TO SOLVE THE PROFILES ------------------*/
/* -------------------------------------------------------------------*/
#include "includes_definitions.h"


// /* Compute the NO consistent solution of the profiles a.k.a forward solver *
// *  INPUTS:                                                                 *    
// *  a (double) : dumping parameter of the profiles                          *
// *  r (double) : line strength vs continium                                 *
// *  eps (double) : photon destruction probavility (how much LTE is)         *
// *  dep_col (double) : effect of depolaraizing colisions                    *
// *  Hd (double) : effect of the Hanle effect                                *
// *
// *  INPUTS AND OUTPUTS:                                                     *
// *  Jm00_nodes (double[NODES]) : old->new Jm00 radiation field at the nodes *
// *  Jm20_nodes (double[NODES]) : old->new Jm20 radiation field at the nodes *
// *
// *  OUTPUTS:                                                                *
// *  I_res (double[nz][nw][qnd]) : intensity of each ray/wavelength/height   *
// *  Q_res (double[nz][nw][qnd]) : Q of each ray/wavelength/height           *
void solve_profiles(double a,double r, double eps, double dep_col, double Hd,
                    double I_res[nz][nw][qnd], double Q_res[nz][nw][qnd],
                    double Jm00_nodes[NODES], double Jm20_nodes[NODES]){

    int i,j,k,l,m;               /* define the integers to count the loops*/

    double II[nz][nw][qnd], QQ[nz][nw][qnd];
    double S00[nz], S20[nz];
    double SI[nz][nw][qnd], SQ[nz][nw][qnd];
    double SLI[nz][qnd], SLQ[nz][qnd];
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
    }

    // initialice the Jm00 and Jm20 with the interpolation of the nodes
    for (i=0; i<nz; i++){
        Jm00[i] = Jm00_nodes[i];
        Jm20[i] = Jm00_nodes[i];
    }
    
    w2jujl = 1.;
    gauleg(-1, 1, mus, weigths, qnd);
    
    for (i=0; i<nz; i++){
        S00[i] = (1-eps)*Jm00[i] + eps;
        S20[i] = Hd * (1-eps)/(1 + (1-eps)*dep_col) * w2jujl * Jm20[i];

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

    /*--------------------------- SOLVE THE RTE --------------------------*/  
    RTE_SC_solve(II, QQ, SI, SQ, taus, mus);

    /* Check for negative intensities to stop and report a problem */
    for (i = 0; i < nz; i++){ 
        for (j = 0; j < nw; j++){
            for (k = 0; k < qnd; k++){
                if(II[i][j][k] < -1e-4){
                    fprintf(stdout,"forward_solver.c found a negative intensity, stopping\n" \
                        "Bad parameters:\n a = %1.8e\n r = %1.8e\n eps = %1.8e\n delta = %1.8e\n" \
                        "Hd = %1.8e\n", a, r, eps, dep_col, Hd);
                    // exit(0);
                    goto wrapup;
                }
            }
        }
    }

    /* -------------------      COMPUTE THE NEW J    -----------------------*/

    for (i = 0; i < nz; i++){
        for (j = 0; j < nw; j++){
            for (k = 0; k < qnd; k++){ integrand_mu[k] = (3.*mus[k]*mus[k] - 1)*II[i][j][k] +\
                                        3.*(mus[k]*mus[k] - 1)*QQ[i][j][k]; }
            
            J00[i][j] = num_gaus_quad( II[i][j], weigths, qnd);
            J20[i][j] = num_gaus_quad( integrand_mu, weigths, qnd);
        }

        for (j = 0; j < nw; j++){ integrand_w[j] = phy[j]*J00[i][j];}
        Jm00[i] = 1./2. * trapezoidal(integrand_w, ww, nw);

        for (j = 0; j < nw; j++){ integrand_w[j] = phy[j]*J20[i][j];}
        Jm20[i] = 1./(4.*sqrt(2)) * trapezoidal(integrand_w, ww, nw);
    }

    wrapup:

    // printf("finished in %i iterations, with tolerance:  %1.2e \n",l, mrc);

    for (i = 0; i < nz; i++){
        for (j = 0; j < nw; j++){
            for (k = 0; k < qnd; k++){
                I_res[i][j][k] = II[i][j][k];
                Q_res[i][j][k] = QQ[i][j][k];
            }
        }
    }

    // Retrieve back the info about the Jms to the nodes
    for (i=0; i<NODES; i++){
        Jm00_nodes[i] = Jm00[i];
        Jm20_nodes[i] = Jm20[i];
    }

    return;
}