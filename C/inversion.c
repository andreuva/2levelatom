/***************************************************************
*        2 LEVEL ATOM ATMOSPHERE SOLVER                        *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
****************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "integratives.h"
#include "params.h"
#include "subroutines.c"
#include "forward_solver.c"

int main(){

    double II[nz][nw][qnd], QQ[nz][nw][qnd];

    solve_profiles( a, r, eps, dep_col, Hd, II, QQ);

    return 0;
}

