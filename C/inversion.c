/***************************************************************
*        2 LEVEL ATOM ATMOSPHERE SOLVER                        *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
****************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include "integratives.h"
#include "params.h"
#include "subroutines.c"
#include "forward_solver.c"

double* generate(int n)
{
    int i;
    int m = n + n % 2;
    double* values = (double*)calloc(m,sizeof(double));
    double average, deviation;
 
    if ( values )
    {
        for ( i = 0; i < m; i += 2 )
        {
            double x,y,rsq,f;
            do {
                x = 2.0 * rand() / (double)RAND_MAX - 1.0;
                y = 2.0 * rand() / (double)RAND_MAX - 1.0;
                rsq = x * x + y * y;
            }while( rsq >= 1. || rsq == 0. );
            f = sqrt( -2.0 * log(rsq) / rsq );
            values[i]   = x * f;
            values[i+1] = y * f;
        }
    }
    return values;
}

void add_noise(double array[nz][nw][qnd], double std){
    
    int i, j, k, index;
    double* normal_noise;

    // initialice the random number generator
    srand((unsigned int)time(NULL));
    normal_noise = generate(nz*nw*qnd);

    for (i = 0; i < nz; i++){
        for (j = 0; j < nw; j++){
            for (k = 0; k < qnd; k++){
                index = i*nw*qnd + j*qnd + k ;
                normal_noise[ index ] = normal_noise[index] * std;
                array[i][j][k] = array[i][j][k] + normal_noise[index];

            }
        }
    }
    return;
}


int main(){

    double I_sol[nz][nw][qnd], Q_sol[nz][nw][qnd];
    double std_I = 1e-6, std_Q = 1e-6; 
    int i, j, k;

    solve_profiles( a, r, eps, dep_col, Hd, I_sol, Q_sol);
    
    add_noise(I_sol, std_I);
    add_noise(Q_sol, std_Q);

    return 0;
}

