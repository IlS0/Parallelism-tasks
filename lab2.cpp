#include <iostream>
#include <cmath>

//поддержка double
//#define LF_SUP
#ifdef LF_SUP 
#define TYPE double
#define ABS fabs
#else
#define TYPE float
#define ABS fabsf
#endif


int main(){

    TYPE error {0.0},tol{-1e-6};
    int iter_max{1000000},iter{0},n{128},m{128};


    TYPE** A = new TYPE*[m]; 
    TYPE** Anew = new TYPE*[m]; 
    for ( int i {0}; i < m; ++i){
        A[i] = new TYPE[n];
        Anew[i] = new TYPE[n];
    }

    A[0][0]=10;
    A[0][m-1] = 20;
    A[n-1][0]=30;
    A[n-1][m-1]=20;



    #pragma acc data copy(A [1:n] [1:m]) create(Anew[n][m])
    while (error > tol && iter < iter_max){
        error = 0.0;

        #pragma acc parallel loop reduction(max : error)
        for (int j {1}; j < n - 1; ++j){
            #pragma acc loop reduction(max : error)
            for (int i {1}; i < m - 1; ++i){
                Anew[j][i] = 0.25 * (A[j][i + 1] + A[j][i - 1] + A[j - 1][i] + A[j + 1][i]);
                error = fmax(error, fabs(Anew[j][i] - A[j][i]));
            }
        }

        #pragma acc parallel loop
        for (int j {1}; j < n - 1; ++j){
            #pragma acc loop
            for (int i {1}; i < m - 1; ++i){
                A[j][i] = Anew[j][i];
            }
        }

        if (iter % 100 == 0){
            printf("%5d, %0.6f\n", iter, error);
        }
        iter++;
    }

    std::cout<<"Iterations: "<<iter<<std::endl<<"Error: "<<error<<std::endl;

    for (int i {0}; i < m; i++){
        delete[] A[i];
        delete[] Anew[i];
    }
    delete [] A;
    delete [] Anew;
}

