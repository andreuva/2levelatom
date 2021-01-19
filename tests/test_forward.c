#include "forward_solver.c"


int main(){
    double II[nz][nw][qnd], QQ[nz][nw][qnd];

    double a = 1e-12;                      /* # dumping Voigt profile a=gam/(2^1/2*sig) */
    double r = 1e-12;                     /* # line strength XCI/XLI */
    double eps = 1e-4;                    /* # Phot. dest. probability (LTE=1,NLTE=1e-4) */
    double dep_col = 0.5;                   /* # Depolirarization colisions (delta) */
    double Hd = 0.20;                        /* # Hanle depolarization factor [1/5, 1] */

    a = 1.00000000e+00;
    r = 1.00000000e-12;
    eps = 1.00000000e-04;
    dep_col = 0.00000000e+00;
    Hd = 1.00000000e+00;

    fprintf(stdout,"\nforward_solver.c test to check the parameters:\n" \
                    " a = %1.8e\n r = %1.8e\n eps = %1.8e\n delta = %1.8e\n" \
                    " Hd = %1.8e\n\n", a, r, eps, dep_col, Hd);
    solve_profiles(a, r, eps, dep_col, Hd, II, QQ);
    return 0;
}