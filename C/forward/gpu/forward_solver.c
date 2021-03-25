#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

/* ------- DEFINE SOME OF THE CONSTANTS OF THE PROBLEM ---------- */
#define c 299792458                   /* # m/s */
#define hh 6.626070150e-34             /* # J/s */
#define kb 1.380649e-23                /* # J/K */
#define R 8.31446261815324            /* # J/K/mol */
#define T 5778                    /* # T (isotermic) of the medium */

/* ------ DEFINE THE PROBLEM PARAMETERS ------- */
#define zl -15.0           /* optical thicknes of the lower boundary */
#define zu 9.0           /* optical thicknes of the upper boundary */
#define dz 0.75                         /* (zu-zl)/(nz-1) */

#define wl -10.0          /* # lower/upper frequency limit (lambda in nm) */
#define wu 10.0                         /*c/(498e-9) */
#define dw 0.25                       /* (wu-wl)/(nw-1) */

#define qnd 14                        /* # nodes in the gaussian quadrature (# dirs) (odd) */

#define ju 1
#define jl 0

#define tolerance 1e-10           /* # Tolerance for finding the solution */
#define max_iter 500              /* maximum number of iterations */
#define EPS 3.0e-16              /* EPS is the relative precision for integrations. */

/*const double w0   c/(500e-9)*/           /* wavelength of the transition (nm --> hz) */

const short nw = (wu-wl)/dw + 1;                /* # points to sample the spectrum */
const short nz = (zu-zl)/dz + 1;                        /* # number of points in the z axes */

/*******************************************************************************
Gauss-Legendre integration function, gauleg, from "Numerical Recipes in C"
(Cambridge Univ. Press) by W.H. Press, S.A. Teukolsky, W.T. Vetterling, and
B.P. Flannery
*******************************************************************************/
void gauleg(double x1, double x2, double x[], double w[], int n)
/*******************************************************************************
Given the lower and upper limits of integration x1 and x2, and given n, this
routine returns arrays x[1..n] and w[1..n] of length n, containing the abscissas
and weights of the Gauss-Legendre n-point quadrature formula.
*******************************************************************************/
{
	int m,j,i;
	double z1,z,xm,xl,pp,p3,p2,p1;
	m=(n+1)/2; /* The roots are symmetric, so we only find half of them. */
	xm=0.5*(x2+x1);
	xl=0.5*(x2-x1);
	for (i=1;i<=m;i++) { /* Loop over the desired roots. */
		z=cos(3.141592654*(i-0.25)/(n+0.5));
		/* Starting with the above approximation to the ith root, we enter */
		/* the main loop of refinement by Newton's method.                 */
		do {
			p1=1.0;
			p2=0.0;
			for (j=1;j<=n;j++) { /* Recurrence to get Legendre polynomial. */
				p3=p2;
				p2=p1;
				p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
			}
			/* p1 is now the desired Legendre polynomial. We next compute */
			/* pp, its derivative, by a standard relation involving also  */
			/* p2, the polynomial of one lower order.                     */
			pp=n*(z*p1-p2)/(z*z-1.0);
			z1=z;
			z=z1-p1/pp; /* Newton's method. */
		} while (fabs(z-z1) > EPS);
		x[i-1]=xm-xl*z;      /* Scale the root to the desired interval, */
		x[n-i]=xm+xl*z;  /* and put in its symmetric counterpart.   */
		w[i-1]=2.0*xl/((1.0-z*z)*pp*pp); /* Compute the weight             */
		w[n-i]=w[i-1];                 /* and its symmetric counterpart. */
	}
}


/*******************************************************************************
Integrate a 1D array by the gaussian quadrature given the weights and the 
array of points in the quadrature abcises + dimension of the array
Author: Andres Vicente Arevalo 
*******************************************************************************/
double num_gaus_quad(double y[], double weigths[], int nn){
    double result = 0.;

    for (int i = 0; i < nn; i++){
        result = result + y[i]*weigths[i];
    }

    return result;    
}


/*******************************************************************************
Integrate a 1D array by the composite trapezoidal rule
Author: Andres Vicente Arevalo 
*******************************************************************************/
double trapezoidal(double y[], double x[], int n){
    double Integral = 0.;

    for (int i = 1; i < n; i++){
        Integral = Integral + (y[i-1] + y[i])*(x[i]-x[i-1])/2.;
    }
    return Integral;
}

/**********************************************************************/
/*  Subroutines to solve the forward modeling for the SC method       */
/*  Author: Andres Vicente Arevalo      Date: 20-11-2020              */
/**********************************************************************/

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
              double psim[], double psio[], double psip[], int mode){
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
                 double SQ[nz][nw][qnd], double lambda[][nw][qnd], double tau[nz][nw], double mu[qnd]){
    
    // Compute the new intensities form the source function with the SC method
    
    double psip_prev[nw], psim[nw], psio[nw], psip[nw];
    double deltaum[nw], deltaup[nw];
    int i,j,k;

    for (k = 0; k < qnd; k++){
        if (mu[k] > 0){
            
            for (j = 0; j < nw; j++){ psip_prev[j] = 0; }

            for (i = 1; i < nz; i++){
                for (j = 0; j < nw; j++){ deltaum[j] = fabs((tau[i-1][j]-tau[i][j])/mu[k]); }
                
                if (i < nz-1){
                    for (j = 0; j < nw; j++){ deltaup[j] = fabs((tau[i][j]-tau[i+1][j])/mu[k]); }
                    psi_calc(deltaum, deltaup, psim, psio, psip, 2);

                    for (j = 0; j < nw; j++){
                        II[i][j][k] = II[i-1][j][k]*exp(-deltaum[j]) + SI[i-1][j][k]*psim[j] + SI[i][j][k]*psio[j] + SI[i+1][j][k]*psip[j];
                        QQ[i][j][k] = QQ[i-1][j][k]*exp(-deltaum[j]) + SQ[i-1][j][k]*psim[j] + SQ[i][j][k]*psio[j] + SQ[i+1][j][k]*psip[j];
                        lambda[i][j][k] = psip_prev[j]*exp(-deltaum[j]) + psio[j];
                        psip_prev[j] = psip[j];
                    }
                    
                }
                else{
                    psi_calc(deltaum, deltaup, psim, psio, psip, 1);
                    for (j = 0; j < nw; j++){
                        II[i][j][k] = II[i-1][j][k]*exp(-deltaum[j]) + SI[i-1][j][k]*psim[j] + SI[i][j][k]*psio[j];
                        QQ[i][j][k] = QQ[i-1][j][k]*exp(-deltaum[j]) + SQ[i-1][j][k]*psim[j] + SQ[i][j][k]*psio[j];
                        lambda[i][j][k] = psip_prev[j]*exp(-deltaum[j]) + psio[j];
                    }
                }
                             
            }
        }
        else{

            for (j = 0; j < nw; j++){ psip_prev[j] = 0; }

            for (i = nz-2; i >= 0; i--){
                
                for (j = 0; j < nw; j++){ deltaum[j] = fabs((tau[i][j]-tau[i+1][j])/mu[k]); }
                
                if (i > 0){

                    for (j = 0; j < nw; j++){ deltaup[j] = fabs((tau[i-1][j]-tau[i][j])/mu[k]); }
                    psi_calc(deltaum, deltaup, psim, psio, psip, 2);

                    for (j = 0; j < nw; j++){
                        II[i][j][k] = II[i+1][j][k]*exp(-deltaum[j]) + SI[i+1][j][k]*psim[j] + SI[i][j][k]*psio[j] + SI[i-1][j][k]*psip[j];
                        QQ[i][j][k] = QQ[i+1][j][k]*exp(-deltaum[j]) + SQ[i+1][j][k]*psim[j] + SQ[i][j][k]*psio[j] + SQ[i-1][j][k]*psip[j];
                        lambda[i][j][k] = psip_prev[j]*exp(-deltaum[j]) + psio[j];
                        psip_prev[j] = psip[j];
                    }
                    
                }
                else{
                    psi_calc(deltaum, deltaup, psim, psio, psip, 1);

                    for (j = 0; j < nw; j++){
                        II[i][j][k] = II[i+1][j][k]*exp(-deltaum[j]) + SI[i+1][j][k]*psim[j] + SI[i][j][k]*psio[j];
                        QQ[i][j][k] = QQ[i+1][j][k]*exp(-deltaum[j]) + SQ[i+1][j][k]*psim[j] + SQ[i][j][k]*psio[j];
                        lambda[i][j][k] = psip_prev[j]*exp(-deltaum[j]) + psio[j];
                    }
                }            
            }
        }
    }
    return;
}

/***************************************************************
*        2 LEVEL ATOM ATMOSPHERE SOLVER MAIN PROGRAM           *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
****************************************************************/

const float a = 1;                      /* # dumping Voigt profile a=gam/(2^1/2*sig) */
const float r = 1;                     /* # line strength XCI/XLI */
const float eps = 1e-4;                    /* # Phot. dest. probability (LTE=1,NLTE=1e-4) */
const float dep_col = 1;                   /* # Depolirarization colisions (delta) */
const float Hd = 0.20;                        /* # Hanle depolarization factor [1/5, 1] */

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
