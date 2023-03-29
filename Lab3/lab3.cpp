#include <iostream>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include "cublas_v2.h"
//#include <openacc.h>

// поддержка double
#define LF_SUP

#ifdef LF_SUP
#define TYPE double
#define ABS fabs
#define MAX fmax
#define CAST std::stod
#else
#define TYPE floatпш
#define ABS fabsf
#define MAX fmaxf
#define CAST std::stof
#endif


#define IDX2C(i, j, ld) (((j)*(ld))+(i))


// инициализация сетки
void initArr(TYPE *A, int n)
{
    #pragma acc kernels
    {
        A[IDX2C(0, 0, n)] = 10.0;
        A[IDX2C(0, n - 1, n)] = 20.0;
        A[IDX2C(n - 1, 0, n)] = 20.0;
        A[IDX2C(n - 1, n - 1, n)] = 30.0;
    }

    
    #pragma acc parallel loop present(A)
    for (int i{1}; i < n - 1; ++i)
    {
        A[IDX2C(0,i,n)] = 10 + (i * 10.0 / (n - 1));
        A[IDX2C(i,0,n)] = 10 + (i * 10.0 / (n - 1));
        A[IDX2C(n-1,i,n)] = 20 + (i * 10.0 / (n - 1));
        A[IDX2C(i,n-1,n)] = 20 + (i * 10.0 / (n - 1));
    }
}

 void printArr(TYPE *A, int n)
{
    for (int i {0}; i < n; ++i)
    {
        for (int j {0}; j < n; ++j)
        {
            #pragma acc kernels present (A)
            printf("%lf ", A[IDX2C(i,j,n)]);
        }
        std::cout<<std::endl;
    }
    
}


void solution(TYPE tol, int iter_max, int n)
{
    //acc_set_device_num(3,acc_device_default);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t status;

    TYPE error{1.0};
    int iter{0},size{n*n};
    
    TYPE alpha {-1};
    int inc {1}, max_idx { 0};

    TYPE *A = new TYPE [size], *Anew = new TYPE [size], *Adif = new TYPE [size];
    
    bool flag {true};

    #pragma acc enter data copyin(error) create(A[0:size],Anew [0:size], Adif[0:size])

    initArr(A, n);
    initArr(Anew, n);

    //printArr(A,n);
    //std::cout<<"___________________________"<<std::endl;
    
    while ( iter < iter_max)
    {
        flag = !(iter % n);

        if(flag){
            #pragma acc kernels present(error)
            error = 0; 
        }

        #pragma acc kernels loop independent collapse(2) present(A, Anew) async(1)
        for (int j{1}; j < n - 1; ++j){
            for (int i{1}; i < n - 1; ++i){
                Anew[IDX2C(j, i, n)] = 0.25 * (A[IDX2C(j, i+1, n)] + A[IDX2C(j, i-1, n)] + A[IDX2C(j-1, i, n)] + A[IDX2C(j+1, i, n)]);
            }
        }

        // swap без цикла
        TYPE *temp = A;
        A = Anew;
        Anew = temp;

        if(flag){  
            #pragma acc data present(A, Anew, Adif) wait(1)
            {
                #pragma acc host_data use_device(A, Anew, Adif)
                {
                    status = cublasDcopy(handle, size, Anew, inc, Adif, inc);
                    if(status != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "copy error" << std::endl; 
                        exit(30);
                    }

                    status = cublasDaxpy(handle, size, &alpha, A, inc, Adif, inc);
                    if(status != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "sum error" << std::endl;
                        exit(40);
                    }
                    
                    status = cublasIdamax(handle, size, Adif, inc, &max_idx);
                    if(status != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "abs max error" << std::endl; 
                        exit(41);
                    }
                
                    //std::cout<<error<<std::endl;
                    #pragma acc kernels present(error)
                    error = ABS(Adif[max_idx - 1]);	
                    
                }   
            }

            #pragma acc update host(error)
            if (error <= tol){
                break;
            }
        }
        
        
        ++iter;   
    }

    #pragma acc wait(1)


    std::cout << "Iterations: " << iter << std::endl<< "Error: " << error << std::endl;
    #pragma acc exit data delete (A [0:size], Anew [0:size], Adif[0:size],error)

    cublasDestroy(handle);
    delete[] A;
    delete[] Anew;
    delete[] Adif;
}

int main(int argc, char *argv[])
{
    
    TYPE tol{1e-6};
    int iter_max{1000000}, n{128}; // значения для отладки, по умолчанию инициализировать нулями

    std::string tmpStr;
    //-t - точность
    //-n - размер сетки
    //-i - кол-во итераций
    for (int i{1}; i < argc; ++i)
    {
        tmpStr = argv[i];
        if (!tmpStr.compare("-t"))
        {
            tol = CAST(argv[i + 1]);
            ++i;
        }

        if (!tmpStr.compare("-i"))
        {
            iter_max = std::stoi(argv[i + 1]);
            ++i;
        }

        if (!tmpStr.compare("-n"))
        {
            n = std::stoi(argv[i + 1]);
            ++i;
        }
    }

    solution(tol, iter_max, n);
}
