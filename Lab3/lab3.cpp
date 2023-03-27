#include <iostream>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// поддержка double
#define LF_SUP

#ifdef LF_SUP
#define TYPE double
#define ABS fabs
#define MAX fmax
#define CAST std::stod
#else
#define TYPE float
#define ABS fabsf
#define MAX fmaxf
#define CAST std::stof
#endif

// инициализация сетки
void initArr(TYPE **A, int n)
{
    #pragma acc kernels
    {
        A[0][0] = 10.0;
        A[0][n - 1] = 20.0;
        A[n - 1][0] = 20.0;
        A[n - 1][n-1] = 30.0;
    }

    #pragma acc parallel loop present(A [0:n] [0:n])
    for (int i{1}; i < n - 1; ++i)
    {
        A[0][i] = 10 + (i * 10.0 / (n - 1));
        A[i][0] = 10 + (i * 10.0 / (n - 1));
        A[n - 1][i] = 20 + (i * 10.0 / (n - 1));
        A[i][n - 1] = 20 + (i * 10.0 / (n - 1));
    }
}

void printArr(TYPE **A, int n)
{
    for (int i {0}; i < n; ++i)
    {
        for (int j {0}; j < n; ++j)
        {
            #pragma acc kernels present (A[0:n][0:n])
            printf("%lf ", A[i][j]);
        }
        std::cout<<std::endl;
    }
    
}

void solution(TYPE tol, int iter_max, int n)
{
    TYPE error{1.0};
    int iter{0};
    bool flag {true};

    TYPE **A = new TYPE *[n], **Anew = new TYPE *[n];
    for (int i{0}; i < n; ++i)
    {
        A[i] = new TYPE[n];
        Anew[i] = new TYPE[n];
    }

    #pragma acc enter data copyin(A [0:n] [0:n], error) create(Anew [0:n] [0:n])

    initArr(A, n);
    initArr(Anew, n);
    //printArr(Anew,n);

    while ( iter < iter_max)
    {
        flag = !(iter % n);

        if (flag){
            #pragma acc kernels present(error)
            error = 0;
            //#pragma acc update device(error)
        }

        #pragma acc parallel loop collapse(2) present(A, Anew, error) reduction(max: error) async(1)
        for (int j{1}; j < n - 1; ++j)
        {
            for (int i{1}; i < n - 1; ++i)
            {
                Anew[j][i] = 0.25 * (A[j][i + 1] + A[j][i - 1] + A[j - 1][i] + A[j + 1][i]);
                if (flag)
                    error = MAX(error, ABS(Anew[j][i] - A[j][i]));
            }
        }

        // swap без цикла
        TYPE **temp = A;
        A = Anew;
        Anew = temp;

        ++iter;
        if (flag){
            #pragma acc update host(error) wait(1)
            if (error <= tol)
                break;
        }
    }
    #pragma acc wait(1)

    #pragma acc exit data delete (A [0:n] [0:n], error, Anew [0:n] [0:n])

    std::cout << "Iterations: " << iter << std::endl<< "Error: " << error << std::endl;

    for (int i{0}; i < n; i++)
    {
        delete[] A[i];
        delete[] Anew[i];
    }
    delete[] A;
    delete[] Anew;
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
