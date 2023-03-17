#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10000000
#define PI 3.14159

//поддержка double
#define LF_SUP
#ifdef LF_SUP 
#define TYPE double
#define SINUS sin
#else
#define TYPE float
#define SINUS sinf
#endif


int main(){
    TYPE *arr = (TYPE*)malloc(sizeof(TYPE) * N), sum =0.0,
    tmp=(PI*2/N);
     
    #pragma acc enter data create(arr[:N]) copyin(tmp,sum)

    #pragma acc kernels
    for (int i = 0; i < N; ++i){ 
        arr[i] = SINUS(tmp * i);
    }


    #pragma acc parallel loop reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += arr[i];
    }

    #pragma acc exit data copyout(sum)
    printf("%-32.25lf\n", sum);
    free(arr);
    return 0;
}