/***************************************************************
*    2 LEVEL ATOM ATMOSPHERE SOLVER CUDA ROUTINES              *
*         AUTHOR: ANDRES VICENTE AREVALO                       *
*    Compilation: nvcc -c routines_gpu.cu -o routines_gpu.o    *
****************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_extension.h"
#include <cuda_runtime.h>

/* Function to compute the coeficients of the SC method for lineal and quadratic */
__device__ void psi_calc_kernel(double deltaum[], double deltaup[], \
              double psim[], double psio[], double psip[], int mode){
    
    // Compute of the psi coefficients in the SC method
    double U0[nw], U1[nw], U2[nw];
    int j;

    for (j = 0; j < nw; j++){
        
        if (deltaum[j] < 1e-3){
            U0[j] = deltaum[j] - deltaum[j]*deltaum[j]/2 +\
                    deltaum[j]*deltaum[j]*deltaum[j]/6;
        }
        else{
            U0[j] = 1 - exp(-deltaum[j]);
        }
        U1[j] = deltaum[j] - U0[j];
    }
     
    
    if (mode == 1){
        for (j = 0; j < nw; j++){
            psim[j] = U0[j] - U1[j]/deltaum[j];
            psio[j] = U1[j]/deltaum[j];
            psip[j] = 0;
        }
    }
    else if (mode == 2){
        for (j = 0; j < nw; j++){
            U2[j] = deltaum[j]*deltaum[j] - 2*U1[j];

            psim[j] = U0[j] + (U2[j] - U1[j]*(deltaup[j] + 2*deltaum[j]))/\
                                        (deltaum[j]*(deltaum[j] + deltaup[j]));
            psio[j] = (U1[j]*(deltaum[j] + deltaup[j]) - U2[j])/(deltaum[j]*deltaup[j]);
            psip[j] = (U2[j] - U1[j]*deltaum[j])/(deltaup[j]*(deltaup[j]+deltaum[j]));
        }        
    }
    else{
        printf("ERROR IN MODE OF THE PSICALC FUNCTION");
    }

    return;
}

__global__ void RTE_SC_kernel(double II[][nw][qnd], double QQ[][nw][qnd], double SI[nz][nw][qnd],\
                 double SQ[nz][nw][qnd], double lambda[][nw][qnd], double tau[nz][nw], double mu[qnd]) {

    // Compute the new intensities form the source function with the SC method
    
    double psip_prev[nw], psim[nw], psio[nw], psip[nw];
    double deltaum[nw], deltaup[nw];
    int i,j,k;

    k = threadIdx.x;
    
    if (mu[k] > 0){
        
        for (j = 0; j < nw; j++){ psip_prev[j] = 0; }

        for (i = 1; i < nz; i++){
            for (j = 0; j < nw; j++){ deltaum[j] = fabs((tau[i-1][j]-tau[i][j])/mu[k]); }
            
            if (i < nz-1){
                for (j = 0; j < nw; j++){ deltaup[j] = fabs((tau[i][j]-tau[i+1][j])/mu[k]); }
                psi_calc_kernel(deltaum, deltaup, psim, psio, psip, 2);

                for (j = 0; j < nw; j++){
                    II[i][j][k] = II[i-1][j][k]*exp(-deltaum[j]) + SI[i-1][j][k]*psim[j] + SI[i][j][k]*psio[j] + SI[i+1][j][k]*psip[j];
                    QQ[i][j][k] = QQ[i-1][j][k]*exp(-deltaum[j]) + SQ[i-1][j][k]*psim[j] + SQ[i][j][k]*psio[j] + SQ[i+1][j][k]*psip[j];
                    lambda[i][j][k] = psip_prev[j]*exp(-deltaum[j]) + psio[j];
                    psip_prev[j] = psip[j];
                }
                
            }
            else{
                psi_calc_kernel(deltaum, deltaup, psim, psio, psip, 1);
                for (j = 0; j < nw; j++){
                    II[i][j][k] = II[i-1][j][k]*exp(-deltaum[j]) + SI[i-1][j][k]*psim[j] + SI[i][j][k]*psio[j];
                    QQ[i][j][k] = QQ[i-1][j][k]*exp(-deltaum[j]) + SQ[i-1][j][k]*psim[j] + SQ[i][j][k]*psio[j];
                    lambda[i][j][k] = psip_prev[j]*exp(-deltaum[j]) + psio[j];
                }
            }
                            
        }
    }
    else{

        for (j = 0; j < nw; j++){ psip_prev[j] = 0; }

        for (i = nz-2; i >= 0; i--){
            
            for (j = 0; j < nw; j++){ deltaum[j] = fabs((tau[i][j]-tau[i+1][j])/mu[k]); }
            
            if (i > 0){

                for (j = 0; j < nw; j++){ deltaup[j] = fabs((tau[i-1][j]-tau[i][j])/mu[k]); }
                psi_calc_kernel(deltaum, deltaup, psim, psio, psip, 2);

                for (j = 0; j < nw; j++){
                    II[i][j][k] = II[i+1][j][k]*exp(-deltaum[j]) + SI[i+1][j][k]*psim[j] + SI[i][j][k]*psio[j] + SI[i-1][j][k]*psip[j];
                    QQ[i][j][k] = QQ[i+1][j][k]*exp(-deltaum[j]) + SQ[i+1][j][k]*psim[j] + SQ[i][j][k]*psio[j] + SQ[i-1][j][k]*psip[j];
                    lambda[i][j][k] = psip_prev[j]*exp(-deltaum[j]) + psio[j];
                    psip_prev[j] = psip[j];
                }
                
            }
            else{
                psi_calc_kernel(deltaum, deltaup, psim, psio, psip, 1);

                for (j = 0; j < nw; j++){
                    II[i][j][k] = II[i+1][j][k]*exp(-deltaum[j]) + SI[i+1][j][k]*psim[j] + SI[i][j][k]*psio[j];
                    QQ[i][j][k] = QQ[i+1][j][k]*exp(-deltaum[j]) + SQ[i+1][j][k]*psim[j] + SQ[i][j][k]*psio[j];
                    lambda[i][j][k] = psip_prev[j]*exp(-deltaum[j]) + psio[j];
                }
            }            
        }
    }
    return;
}

void RTE_SC_solve(double II[nz][nw][qnd], double QQ[nz][nw][qnd], double SI[nz][nw][qnd],\
                 double SQ[nz][nw][qnd], double lambda[nz][nw][qnd], double tau[nz][nw], double mu[qnd]) {

    // int i,j,k;

    dim3 thrds_per_block(qnd,1,1);
    dim3 blcks_per_grid(1,1,1);

    RTE_SC_kernel<<< blcks_per_grid, thrds_per_block >>>(II, QQ, SI, SQ, lambda, tau, mu);
    cudaDeviceSynchronize();

    return;
}
