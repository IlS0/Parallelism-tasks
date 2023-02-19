#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>




#define N 10000000
#define PI 3.14159

//поддержка double
#define LF_SUP
#ifdef LF_SUP 
#define TYPE double
#else
#define TYPE float
#endif


int main(){
    struct timespec start, end;   

    clock_gettime(CLOCK_REALTIME, &start);
    TYPE *arr = (TYPE*)malloc(sizeof(TYPE) * N), sum =0.0,
    tmp=(PI*2/N);
     
    #pragma acc enter data create(arr[:N]) copyin(tmp,sum)

    #pragma acc kernels
    for (int i = 0; i < N; ++i){ 
        arr[i] = sin(tmp * i);
    }

    #pragma acc parallel loop reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += arr[i];
    }

    #pragma acc exit data copyout(sum)
    printf("%-32.25lf\n", sum);
    free(arr);

    clock_gettime(CLOCK_REALTIME, &end);
    double time = ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0)*1000;

 
    printf("%lf ms\n", time);
    return 0;
}