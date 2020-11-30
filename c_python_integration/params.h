/* ------- DEFINE SOME OF THE CONSTANTS OF THE PROBLEM ---------- */
const int c = 299792458;                   /* # m/s */
const float h = 6.626070150e-34;             /* # J/s */
const float kb = 1.380649e-23;                /* # J/K */
const float R = 8.31446261815324;            /* # J/K/mol */
const int T = 5778;                    /* # T (isotermic) of the medium */

/* ------ DEFINE THE PROBLEM PARAMETERS ------- */
const float zl = -15; /*-log(1e3);          /* optical thicknes of the lower boundary */
const float zu = 9; /*-log(1e-3);           /* optical thicknes of the upper boundary */
const float dz = 0.75;                         /* (zu-zl)/(nz-1); */
const short nz = (zu-zl)/dz + 1;                        /* # number of points in the z axes */

const float wl = -10; /*c/(502e-9);          /* # lower/upper frequency limit (lambda in nm) */
const float wu = 10;                         /*c/(498e-9);
/*const float w0 =  c/(500e-9);*/           /* wavelength of the transition (nm --> hz) */
const float dw = 0.25;                       /* (wu-wl)/(nw-1); */
const short nw = (wu-wl)/dw + 1;                /* # points to sample the spectrum */

const short qnd = 14;                        /* # nodes in the gaussian quadrature (# dirs) (odd) */

const short ju = 1;
const short jl = 0;

const float tolerance = 1e-10;           /* # Tolerance for finding the solution */
const int max_iter = 500;              /* maximum number of iterations */
