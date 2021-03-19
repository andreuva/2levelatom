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
