#include <stdio.h>
#include <stdlib.h>

//passing 2D array as a parameter to a function
void print_mat(int arr[][4]){
    int i, j, k;

    for (i = 0; i < 3; i++){
        for (j = 0; j < 4; j++){
            printf("%d ", arr[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    return;
}

void modify(int arr[][4]){
    int i,j,k;

    for (i = 0; i < 3; i++){
        for (j = 0; j < 4; j++){
            arr[i][j] = (i+1)*(i+1);
        }        
    }
    fprintf(stdout, "\n");
    return;
}