#include "test_pointers.c"

int main(void){
    int arr1[3][4], arr2[3][4];
    int m = 3, n = 4; //m - no.of rows, n - no.of col
    int i,j, index_1, index_2;
    int *sol;

    fprintf(stdout,"Assigning matrix 1\n");
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            arr1[i][j] = i+j;
            // fprintf(stdout,"trying to asign *arr1[i][j] = i+j : %i  = %i + %i\n", arr1[i][j], i,j);
        }
    }
    fprintf(stdout,"Print matrix 1\n");
    print_mat(arr1);

    fprintf(stdout,"Modifying matrix 1\n");
    modify(arr1);

    fprintf(stdout,"Print modified matrix 1\n");
    print_mat(arr1);

    return 0;
}