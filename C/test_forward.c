#include "forward_solver.c"


int main()
{
    double I_p[nz][nw][qnd], Q_p[nz][nw][qnd];
    for (int i = 0; i < 3; i++)
    {
        solve_profiles( 8.62870272e-03, 2.92362383e-04, 2.71376520e-01, 0.00000000e+00, 2.00000003e-01, I_p, Q_p);        
    }
    
    return 0;
}