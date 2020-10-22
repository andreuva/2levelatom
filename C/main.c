/***************************************************************
*        2 LEVEL ATOM ATMOSPHERE SOLVER                        *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
****************************************************************/

#include <stdio.h>
#include <math.h>

/*  DEFINE SOME OF THE CONSTANTS OF THE PROBLEM  */
const int c = 299792458;                   /* # m/s */
const float h = 6.626070150e-34;             /* # J/s */
const float kb = 1.380649e-23;                /* # J/K */

/*  DEFINE THE PROBLEM PARAMETERS                */
const float zl = 1e0;                    /* optical thicknes of the lower boundary */
const float zu = 1e-3;                   /* optical thicknes of the upper boundary */
const short nz = 20;                    /* # number of points in the z axes */

const float wl = c/(5000e-9);                /* # lower/upper frequency limit (lambda in nm) */
const float wu = c/(350e-9);
const float w0 = c/(1000e-9);                /* wavelength of the transition (nm --> hz) */
const short nw = 25;                        /* # points to sample the spectrum */

const short qnd = 10;                   /* # nodes in the gaussian quadrature (# dirs) */

const int T = 5778;                    /* # T (isotermic) of the medium */

const float a = 10;                      /* # dumping Voigt profile a=gam/(2^1/2*sig) */
const float r = 0.5;                     /* # line strength XCI/XLI */
const float eps = 1e-1;                  /* # Phot. dest. probability (LTE=1,NLTE=1e-4) */
const float dep_col = 0.5;               /* # Depolirarization colisions (delta) */
const float Hd = 0.2;                    /* # Hanle depolarization factor [1/5, 1] */
const short ju = 1;
const short jl = 0;

const float tolerance = 1e-15;           /* # Tolerance for finding the solution */

const float dz = (zl-zu)/(nz-1);
const float wa = c/(1000e-9) * 0.05;
const float dw = (wu-wl)/(nw-1);


int main() {
    

    fprintf(stdout, "\n------------------- PARAMETERS OF THE PROBLEM ---------------------\n");
    fprintf(stdout, "optical thicknes of the lower boundary:            %1.1e \n", zl);
    fprintf(stdout, "optical thicknes of the upper boundary:            %1.1e \n", zu);
    fprintf(stdout, "number of points in the z axes:                    %i \n", nz);
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
    fprintf(stdout, "Tolerance for finding the solution:                %f \n", tolerance);
    fprintf(stdout, "------------------------------------------------------------------\n\n");


    int i,j,k,l,m;               /* define the integers to count the loops*/
    
    double zz[nz];               /* compute the 1D grid in z */
    for(i=0; i<nz; i++){
        zz[i] = zl + i*dz;
    }
    
    double phy[nz][nw][qnd];
    double ww[nw], wnorm[nw], norm=0;               
    /* compute the 1D grid in z, the Voigt profile and the normalization */
    for(i=0; i<nz; i++){
        for(j=0; j<nw; j++){
            ww[j] = wl + j*dw;
            wnorm[j] = (ww[j] - w0)/wa;
            for(k=0; k<qnd; k++){
                phy[i][j][k] = 1; /*voigt(wnorm[i], 1e-9);*/
            }
        }
        if( i > 0 ){ norm += (phy[i][j][k]+ phy[i-1][j][k]) * dw/2; }
    }
    

    for(i=0; i<nz; i++){
        for(j=0; i<nw; i++){
            for(k=0; i<qnd; i++){
                phy[i][j][k] = phy[i][j][k]/norm;
            }
        }
    }

    double mus[qnd];               /* compute the 1D grid in z */
    for(i=0; i<nz; i++){
        mus[i] = -1 + i*2/(qnd-1);
    }

    /*
    ww = np.arange(pm.wl, pm.wu, pm.dw)         /* Compute the 1D spectral grid */
    /*
    mus = np.linspace(-1, 1, pm.qnd)            /* compute the directions */
    /*
    # ------------------------ SOME INITIAL CONDITIONS -----------------------------
    # Compute the initial Voigts vectors
    phy = np.empty_like(ww)
    wnorm = (ww - pm.w0)/pm.wa          # normalice the frequency to compute phy */

















    return 0;
}