#include <iostream>
#include <cmath>
#include <string>
#include <chrono>

//поддержка double
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

//инициализация сетки
void initArr(TYPE** A,int n){
    A[0][0]=10.0;
    A[0][n-1] = 20.0;
    A[n-1][0]=30.0;
    A[n-1][n-1]=20.0;

    #pragma acc parallel loop present(A[0:n][0:n])
    for (int i{1};i<n-1;++i){
        A[0][i]=10+((A[0][n-1] - A[0][0])/ (n-1))*i;
        A[i][0]=10+((A[n-1][0] - A[0][0])/ (n-1))*i;
        A[n-1][i]=10+((A[n-1][0] - A[n-1][n-1])/ (n-1))*i;
        A[i][n-1]=10+((A[n-1][n-1] - A[n-1][0])/ (n-1))*i;
    }
}

void solution(TYPE tol,int iter_max,int n){
    TYPE error {1.0};
    int iter{0};
    TYPE** A = new TYPE*[n],**Anew = new TYPE*[n]; 
    for ( int i {0}; i < n; ++i){
        A[i] = new TYPE[n];
        Anew[i] = new TYPE[n];
    }

    #pragma acc enter data copyin(A [0:n] [0:n],error) create (Anew[0:n][0:n])

    initArr(A,n);
    initArr(Anew,n);
        
    while (error > tol && iter < iter_max){
        error = 0.0;
        #pragma acc update device(error)

        #pragma acc parallel loop collapse(2) present(A[0:n][0:n]) reduction(max : error)
        for (int j {1}; j < n - 1; ++j){
            for (int i {1}; i < n - 1; ++i){
                Anew[j][i] = 0.25 * (A[j][i + 1] + A[j][i - 1] + A[j - 1][i] + A[j + 1][i]);
                error = MAX(error, ABS(Anew[j][i] - A[j][i]));
            }
        }

        //swap без цикла
        TYPE** temp = A;
        A = Anew;
        Anew = temp;    

        ++iter;
        #pragma acc update host(error)
    }

    #pragma acc exit data delete(A [0:n] [0:n],error,Anew [0:n] [0:n])
    
    std::cout<<"Iterations: "<<iter<<std::endl<<"Error: "<<error<<std::endl;

    for (int i {0}; i < n; i++){
        delete[] A[i];
        delete[] Anew[i];
    }
    delete [] A;
    delete [] Anew;
}

int main(int argc, char *argv[]){
    
    TYPE tol{1e-6};
    int iter_max{1000000},n{128}; //значения для отладки, по умолчанию инициализировать нулями

    std::string tmpStr;
    //-t - точность
    //-n - размер сетки
    //-i - кол-во итераций
    for (int i{1};i<argc;++i){
        tmpStr = argv[i];
        if(!tmpStr.compare("-t")){
            tol = CAST(argv[i + 1]);
            ++i;
        }

        if(!tmpStr.compare("-i")) {
            iter_max = std::stoi(argv[i + 1]);
            ++i;
        }

        if(!tmpStr.compare("-n")) {
            n = std::stoi(argv[i + 1]);
            ++i;
        } 
    }
    auto start = std::chrono::high_resolution_clock::now();
    solution(tol,iter_max,n);
    auto end = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end).count();
    std::cout<<"Time (ms): "<<microseconds/1000<<std::endl;
}

