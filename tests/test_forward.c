#include "forward_solver.c"


int main(){
    double II[nz][nw][qnd], QQ[nz][nw][qnd];

    double a = 1e-12;                      /* # dumping Voigt profile a=gam/(2^1/2*sig) */
    double r = 1e-12;                     /* # line strength XCI/XLI */
    double eps = 1e-4;                    /* # Phot. dest. probability (LTE=1,NLTE=1e-4) */
    double dep_col = 0.5;                   /* # Depolirarization colisions (delta) */
    double Hd = 0.20;                        /* # Hanle depolarization factor [1/5, 1] */

    for (int i = 0; i < 8; i++)
    {
        a = a + a*10;
        Hd = Hd + 0.1;
        eps = eps + 3.34e-3;
        r = r + r*10;
        fprintf(stdout,"%1.6e   %1.6e  %1.6e  %1.6e\n",a, r, eps, Hd);
        solve_profiles(a, r, eps, dep_col, Hd, II, QQ);
    }
    return 0;
}