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
const short nz = 24;                      /* # number of points in the z axes */

const float wl = c/(502e-9);                /* # lower/upper frequency limit (lambda in nm) */
const float wu = c/(498e-9);
const float w0 = c/(500e-9);                /* wavelength of the transition (nm --> hz) */
const short nw = 10;                        /* # points to sample the spectrum */

const short qnd = 2;                   /* # nodes in the gaussian quadrature (# dirs) (odd) */

const int T = 5778;                    /* # T (isotermic) of the medium */

const float a = 1;                      /* # dumping Voigt profile a=gam/(2^1/2*sig) */
const float r = 0.001;                     /* # line strength XCI/XLI */
const float eps = 1e-4;                  /* # Phot. dest. probability (LTE=1,NLTE=1e-4) */
const float dep_col = 0.5;               /* # Depolirarization colisions (delta) */
const float Hd = 0.25;                    /* # Hanle depolarization factor [1/5, 1] */
const short ju = 1;
const short jl = 0;

const float tolerance = 1e-3;           /* # Tolerance for finding the solution */
const int max_iter = 5000;              /* maximum number of iterations */

const float dz = (zu-zl)/(nz-1);
const float dw = (wu-wl)/(nw-1);

/* -------------------------------------------------------------------*/
/* -----------------------PHYSICAL FUNCTIONS -------------------------*/
/* -------------------------------------------------------------------*/
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

double plank_wien(double nu, int T){
    /*  Return the Wien aproximation to the plank function */
    return (2*h*nu*nu*nu)/(c*c) * exp( -h*nu/(kb*T) );
}

/* -------------------------------------------------------------------*/
/* ------------------------- MAIN PROGRAM ----------------------------*/
/* -------------------------------------------------------------------*/
int main() {

    fprintf(stdout, "\n------------------- PARAMETERS OF THE PROBLEM ---------------------\n");
    fprintf(stdout, "optical thicknes of the lower boundary:            %1.1e \n", zl);
    fprintf(stdout, "optical thicknes of the upper boundary:            %1.1e \n", zu);
    fprintf(stdout, "resolution in the z axis:                          %1.3e \n", dz);
    fprintf(stdout, "lower/upper frequency limit (Hz):                  %1.3e   %1.3e \n", wl, wu);
    fprintf(stdout, "frequency of the transition (Hz):                  %1.2e \n", w0);
    fprintf(stdout, "number points to sample the spectrum:              %i \n", nw);
    fprintf(stdout, "nodes in the gaussian quadrature (# dirs):         %i \n", qnd);
    fprintf(stdout, "T (isotermic) of the medium:                       %i \n", T);
    fprintf(stdout, "dumping Voigt profile a=gam/(2^1/2*sig):           %f \n", a);
    fprintf(stdout, "line strength XCI/XLI:                             %f \n", r);
    fprintf(stdout, "Phot. dest. probability (LTE=1,NLTE=1e-4):         %f \n", eps);
    fprintf(stdout, "Depolirarization colisions (delta):                %f \n", dep_col);
    fprintf(stdout, "Hanle depolarization factor [1/5, 1]:              %f \n", Hd);
    fprintf(stdout, "angular momentum of the levels (Ju, Jl):           (%i,%i) \n", ju, jl);
    fprintf(stdout, "Tolerance for finding the solution:                %f º/. \n", tolerance*100);
    fprintf(stdout, "------------------------------------------------------------------\n\n");


    int i,j,k,l,m;               /* define the integers to count the loops*/

    double II[nz][nw][qnd], II_new[nz][nw][qnd], QQ[nz][nw][qnd], QQ_new[nz][nw][qnd];
    double SI[nz][nw][qnd], SI_new[nz][nw][qnd], SQ[nz][nw][qnd], SQ_new[nz][nw][qnd];
    double Jm00[nz][nw], Jm02[nz][nw];

    double zz[nz], taus[nz];               /* compute the 1D grid in z */
    double ww[nw], wnorm[nw], phy[nw], plank[nw], norm=0;
    double mus[qnd];
    double wa = w0*(sqrt(2*R*T/1e-3))/c;
    
    double psim,psio,psip, U0,U1,U2, deltaum, deltaup, I1, I2;
    double mrc, diff;

    /* double II[nz][nw][qnd],QQ[nz][nw][qnd],SI[nz][nw][qnd],SQ[nz][nw][qnd];
    /* compute the 1D grid in z, the Voigt profile and the normalization */


    for(i=0; i<nz; i++){
        zz[i] = zl + i*dz;
    }

    for(j=0; j<nw; j++){
        ww[j] = wl + j*dw;
        wnorm[j] = (ww[j] - w0)/wa;
        plank[j] =  plank_wien(ww[j], T);
        phy[j] = creal(voigt(wnorm[j],a));
    }

    for(i=0; i<qnd; i++){
        mus[i] = -1 + i*2./(qnd-1);
    }

    for (i=0; i<nz; i++){
        for (j = 0; j < nw; j++){
            for (k = 0; k < qnd; k++){
                if(i==nz-1){ II[i][j][k] = 0; }
                II[i][j][k] = plank[j];
                QQ[i][j][k] = 0;
                SI[i][j][k] = plank[j];
                SQ[i][j][k] = 0;
            }
        }
    }
    /* -------------------------------------------------------------------*/
    /* ---------------------------- MAIN LOOP ----------------------------*/
    /* -------------------------------------------------------------------*/
    for(l=1; l<=max_iter; l++){         /* loop with the total iterations */

        /* -------------------------------------------------------------------*/
        /*--------------------------- SOLVE THE RTE --------------------------*/
        /* -------------------------------------------------------------------*/
        for (k = 0; k < qnd; k++){          /* loop for all the directions*/
            
            if(mus[k] > 0){                     /*check if the direction is upwards */
                
                for (i = 1; i < nz-1; i++){           /* loop for all the z's*/

                    deltaum = fabs(exp(-zz[i]) - exp(-zz[i-1]));
                    deltaup = fabs(exp(-zz[i+1]) - exp(-zz[i]));

                    U0 = 1 - exp(-deltaum);
                    U1 = deltaum - U0;
                    U2 = deltaum*deltaum - 2*U1;

                    psim = U0 + (U2 - U1*(deltaup + 2*deltaum))/(deltaum*(deltaum + deltaup));
                    psio = (U1*(deltaum + deltaup) - U2)/(deltaum*deltaup);
                    psip = (U2 - U1*deltaum)/(deltaup*(deltaup+deltaum));

                    for (j = 0; j < nw; j++){           /* loop over frequencies*/
                        II_new[i][j][k] = II_new[i-1][j][k]*exp(-deltaum) + SI[i-1][j][k]*psim + SI[i][j][k]*psio + SI[i+1][j][k]*psip;
                        QQ_new[i][j][k] = QQ_new[i-1][j][k]*exp(-deltaum) + SQ[i-1][j][k]*psim + SQ[i][j][k]*psio + SQ[i+1][j][k]*psip;
                    }
                    
                }
            
                /*Last point with linear aproximation*/
                deltaum = fabs(exp(-zz[nz-1])/mus[j] - exp(-zz[nz-2])/mus[j]);

                U0 = 1 - exp(-deltaum);
                U1 = deltaum - U0;

                psim = U0 - U1/deltaum;
                psio = U1/deltaum;

                for (j = 0; j < nw; j++){
                    II_new[nz-1][j][k] = II_new[nz-2][j][k]*exp(-deltaum) + SI[nz-2][j][k]*psim + SI[nz-1][j][k]*psio;
                    QQ_new[nz-1][j][k] = QQ_new[nz-2][j][k]*exp(-deltaum) + SQ[nz-2][j][k]*psim + SQ[nz-1][j][k]*psio;
                }

            }
            else{ /*repeat with the downward rays*/

                for (i = nz-1; i > 0; i--){           /* loop for all the z's*/

                    deltaum = fabs(exp(-zz[i])/mus[j] - exp(-zz[i+1])/mus[j]);
                    deltaup = fabs(exp(-zz[i-1])/mus[j] - exp(-zz[i])/mus[j]);

                    U0 = 1 - exp(-deltaum);
                    U1 = deltaum - U0;
                    U2 = deltaum*deltaum - 2*U1;

                    psim = U0 + (U2 - U1*(deltaup + 2*deltaum))/(deltaum*(deltaum + deltaup));
                    psio = (U1*(deltaum + deltaup) - U2)/(deltaum*deltaup);
                    psip = (U2 - U1*deltaum)/(deltaup*(deltaup+deltaum));

                    for (j = 0; j < nw; j++){           /* loop over frequencies*/
                        II_new[i][j][k] = II_new[i+1][j][k]*exp(-deltaum) + SI[i+1][j][k]*psim + SI[i][j][k]*psio + SI[i-1][j][k]*psip;
                        QQ_new[i][j][k] = QQ_new[i+1][j][k]*exp(-deltaum) + SQ[i+1][j][k]*psim + SQ[i][j][k]*psio + SQ[i-1][j][k]*psip;
                    }
                    
                }

                deltaum = fabs(exp(-zz[0])/mus[j] - exp(-zz[1])/mus[j]);

                U0 = 1 - exp(-deltaum);
                U1 = deltaum - U0;

                psim = U0 - U1/deltaum;
                psio = U1/deltaum;

                for (j = 0; j < nw; j++){
                    II_new[0][j][k] = II_new[1][j][k]*exp(-deltaum) + SI[1][j][k]*psim + SI[0][j][k]*psio;
                    QQ_new[0][j][k] = QQ_new[1][j][k]*exp(-deltaum) + SQ[1][j][k]*psim + SQ[0][j][k]*psio;
                }

            }
        }
        /* -------------------------------------------------------------------*/
        /* ------------------- COMPUTE THE J AND NEW S -----------------------*/
        /* -------------------------------------------------------------------*/
        for (i = 0; i < nz; i++){
            for (j = 0; j < nw; j++){
                I1 = 0;
                I2 = 0;
                for (k = 1; k < qnd; k++){
                    I1 += (II_new[i][j][k-1] + II_new[i][j][k])*(mus[k]-mus[k-1])/2.;
                    I2 += (((3*mus[k-1]*mus[k-1] - 1)*II_new[i][j][k-1] + 3*(mus[k-1]*mus[k-1] - 1)*QQ_new[i][j][k-1]) \
                    + ((3*mus[k]*mus[k] - 1)*II_new[i][j][k] + 3*(mus[k]*mus[k] - 1)*QQ_new[i][j][k]) )\
                    * (mus[k]-mus[k-1])/2.;
                }
                
                Jm00[i][j] = 1/2 * I1;
                Jm02[i][j] = 1/(4*sqrt(2)) * I2;
            }
        }
        
        for ( i = 0; i < nz; i++){
            for ( j = 0; j < nw; j++){
                for (k = 0; k < qnd; k++){
                    SI_new[i][j][k] = (1-eps)*Jm00[i][j] + eps*plank[j];
                    SQ_new[i][j][k] = (1-eps)*Jm02[i][j];
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
                    diff = fabs((SI[i][j][k] - SI_new[i][j][k])/SI_new[i][j][k]);
                    if (diff > mrc){
                        mrc = diff;
                    }
                }
            }
        }
        
        printf("Actual tolerance is :  %1.2e \n",mrc*100);
        if (mrc < tolerance){
            break;
        }
        for (i = 0; i < nz; i++){
            for (j = 0; j < nw; j++){
                for (k = 0; k < qnd; k++){
                    II[i][j][k] = II_new[i][j][k];
                    QQ[i][j][k] = QQ_new[i][j][k];
                    SI[i][j][k] = SI_new[i][j][k];
                    SQ[i][j][k] = SQ_new[i][j][k];
                }   
            }   
        }        
    }

    return 0;
}