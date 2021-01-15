/* ------- DEFINE SOME OF THE CONSTANTS OF THE PROBLEM ---------- */
// const int c = 299792458;                   /* # m/s */
// const float h = 6.626070150e-34;             /* # J/s */
// const float kb = 1.380649e-23;                /* # J/K */
// const float R = 8.31446261815324;            /* # J/K/mol */
// const int T = 5778;                    /* # T (isotermic) of the medium */

/* ------ DEFINE THE PROBLEM PARAMETERS ------- */
const double zl = -15; /*-log(1e3);          /* optical thicknes of the lower boundary */
const double zu = 9; /*-log(1e-3);           /* optical thicknes of the upper boundary */
const double dz = 0.75;                         /* (zu-zl)/(nz-1); */
const short nz = (zu-zl)/dz + 1;                        /* # number of points in the z axes */

const double wl = -10; /*c/(502e-9);          /* # lower/upper frequency limit (lambda in nm) */
const double wu = 10;                         /*c/(498e-9);
/*const float w0 =  c/(500e-9);*/           /* wavelength of the transition (nm --> hz) */
const double dw = 0.25;                       /* (wu-wl)/(nw-1); */
const short nw = (wu-wl)/dw + 1;                /* # points to sample the spectrum */

const short qnd = 14;                        /* # nodes in the gaussian quadrature (# dirs) (odd) */

const short ju = 1;
const short jl = 0;

const double tolerance = 1e-10;           /* # Tolerance for finding the solution */
const int max_iter = 500;              /* maximum number of iterations */

const double a_sol = 1e-4;                      /* # dumping Voigt profile a=gam/(2^1/2*sig) */
const double r_sol = 1e-5;                     /* # line strength XCI/XLI */
const double eps_sol = 1e-3;                    /* # Phot. dest. probability (LTE=1,NLTE=1e-4) */
const double dep_col_sol = 0.5;                   /* # Depolirarization colisions (delta) */
const double Hd_sol = 0.5;                        /* # Hanle depolarization factor [1/5, 1] */