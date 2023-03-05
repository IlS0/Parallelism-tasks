#include <iostream>
#include <cmath>

//поддержка double
#define LF_SUP
#ifdef LF_SUP 
#define TYPE double
#define ABS fabs
#define MAX fmax
#else
#define TYPE float
#define ABS fabsf
#define MAX fmaxf
#endif


int main(){

    TYPE error {1.0},tol{1e-6};
    int iter_max{1000000},iter{0},n{128},m{128};


    TYPE** A = new TYPE*[m]; 
    TYPE** Anew = new TYPE*[m]; 
    for ( int i {0}; i < m; ++i){
        A[i] = new TYPE[n];
        Anew[i] = new TYPE[n];
    }

    #pragma acc enter data copy(A [0:n] [0:m],error) create(Anew[n][m])
        A[0][0]=10;
        A[0][m-1] = 20;
        A[n-1][0]=30;
        A[n-1][m-1]=20;
    

        #pragma acc parallel loop present(A[0:n][0:m])
        for (int i{1};i<n-1;++i){
            A[0][i]=10+((A[0][m-1] - A[0][0])/ (n-1))*i;
            A[i][0]=10+((A[n-1][0] - A[0][0])/ (n-1))*i;
            A[n-1][i]=10+((A[n-1][0] - A[n-1][m-1])/ (n-1))*i;
            A[i][m-1]=10+((A[n-1][m-1] - A[n-1][0])/ (n-1))*i;
        }
        
        while (error > tol && iter < iter_max){
            error = 0.0;
            #pragma update device(error)

            #pragma acc parallel loop  present(A[0:n][0:m]) // collapse(2) reduction(max : error)
            for (int j {1}; j < n - 1; ++j){
                #pragma acc loop// reduction(max : error)
                for (int i {1}; i < m - 1; ++i){
                    Anew[j][i] = 0.25 * (A[j][i + 1] + A[j][i - 1] + A[j - 1][i] + A[j + 1][i]);
                    error = MAX(error, fabs(Anew[j][i] - A[j][i]));
                }
            }

            //#pragma acc parallel loop collapse(2)//мб коллапс?
            for (int j {1}; j < n - 1; ++j){
                //#pragma acc loop
                for (int i {1}; i < m - 1; ++i){
                    A[j][i] = Anew[j][i];
                }
            }

            /*if (iter % 100 == 0){
                printf("%5d, %0.6f\n", iter, error);
            }*/
            ++iter;
            #pragma update host(error)
        }
    #pragma acc exit data delete(A [0:n] [0:m],error,Anew[n][m])

    std::cout<<"Iterations: "<<iter<<std::endl<<"Error: "<<error<<std::endl;


    for (int i {0}; i < m; i++){
        delete[] A[i];
        delete[] Anew[i];
    }
    delete [] A;
    delete [] Anew;
}

