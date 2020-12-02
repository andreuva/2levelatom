#include <stdio.h>
#include "linalg.c"

#define numpar 3

void solve_linear_sistem(double aa[numpar][numpar], double bb[numpar], double res[numpar]) {
    int i, j, n = numpar, * indx;
    double d, * b, ** a;
    // initialize matrix  
    a = dmatrix(1, n, 1, n);
    // initialize vector  
    b = dvector(1, n);
    
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            a[i+1][j+1] = aa[i][j];
        }
        b[i+1] = bb[i];
    }
    
    
    
    indx = ivector(1, n);
    // printf("original\n");
    // for (i = 1; i <= n; i++) {
    //     for (j = 1; j <= n; j++)
    //         printf("%f ", a[i][j]);
    //     printf(" %5s  %f   \n", (i == 2 ? "* X =" : "     "), b[i]);
    // }
    ludcmp(a, n, indx, &d);
    // printf("after ludcmp\n");
    // for (i = 1; i <= n; i++) {
    //     for (j = 1; j <= n; j++)
    //         printf("%f ", a[i][j]);
    //     printf(" %5s  %f   \n", (i == 2 ? "* X =" : "     "), b[i]);
    // }
    lubksb(a, n, indx, b);
    // printf("after lubksb\n");
    // for (i = 1; i <= n; i++) {
    //     for (j = 1; j <= n; j++)
    //         printf("%f ", a[i][j]);
    //     printf(" %5s  %f   \n", (i == 2 ? "* X =" : "     "), b[i]);
    // } // calculate determinant  
    // for (j = 1; j <= n; j++) d *= a[j][j];
    // printf("det(A) = %f \n", d);


    for (i = 0; i < n; i++){
        res[i] = b[i+1];
    }

    free_dvector(b, 1, n);
    free_ivector(indx, 1, n);
    free_dmatrix(a, 1, n, 1, n);

    return;
}


int main() {
    double aa[numpar][numpar], bb[numpar], result[numpar];
    int i,j;

    aa[0][0] = 3; aa[0][1] = -2; aa[0][2] = 1;
    aa[1][0] = 1; aa[1][1] = 3;  aa[1][2] = 1;
    aa[2][0] = 2; aa[2][1] = 2;  aa[2][2] = -4;

    bb[0] = 12; bb[1] = -4; bb[2] = 6;     
    
    printf("Original before function:\n");
    for (i = 0; i < numpar; i++) {
        for (j = 0; j < numpar; j++)
            printf("%1.3e ", aa[i][j]);
        printf(" %5s  %1.3e   \n", (i == 1 ? "* X =" : "     "), bb[i]);
    }
    
    solve_linear_sistem(aa, bb, result);

    printf("Original after function:\n");
    for (i = 0; i < numpar; i++) {
        for (j = 0; j < numpar; j++)
            printf("%1.3e ", aa[i][j]);
        printf(" %5s  %1.3e   \n", (i == 1 ? "* X =" : "     "), bb[i]);
    }

    printf("Result:\n");
    for (i = 0; i < numpar; i++) {
        printf("%1.3e ", result[i]);
    }
    printf("\n -------  FINISHED    ------------\n");

}