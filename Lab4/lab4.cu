#include <iostream>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda.h> 
#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh> 


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

//индексация по фортрану
#define IDX2C(i, j, ld) (((j)*(ld))+(i))


// функция инициализации сетки
void initArr(TYPE *A, const int n)
{
    //заполнение углов сетки
    A[IDX2C(0, 0, n)] = 10.0;
    A[IDX2C(0, n - 1, n)] = 20.0;
    A[IDX2C(n - 1, 0, n)] = 20.0;
    A[IDX2C(n - 1, n - 1, n)] = 30.0;


    //заполнение краёв сетки
    for (int i{1}; i < n - 1; ++i)
    {
        A[IDX2C(0,i,n)] = 10 + (i * 10.0 / (n - 1));
        A[IDX2C(i,0,n)] = 10 + (i * 10.0 / (n - 1));
        A[IDX2C(n-1,i,n)] = 20 + (i * 10.0 / (n - 1));
        A[IDX2C(i,n-1,n)] = 20 + (i * 10.0 / (n - 1));
    }
}

//функция печати массива
 void printArr(TYPE *A, const int n)
{
    for (int i {0}; i < n; ++i)
    {
        for (int j {0}; j < n; ++j)
        {

            printf("%lf ", A[IDX2C(i,j,n)]);
        }
        std::cout<<std::endl;
    }
    
}


// Шаг алгоритма
__global__ void Step(const double* A, double* Anew, int* dev_n){
    //вычисление ячейки
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    //проверка границ
    if (j == 0 || i == 0 || i == *dev_n-1 || j == *dev_n-1) return;
    //среднее по соседним элементам
    Anew[IDX2C(j, i, *dev_n)] = 0.25 * (A[IDX2C(j, i+1, *dev_n)] + A[IDX2C(j, i-1, *dev_n)] + A[IDX2C(j-1, i, *dev_n)] + A[IDX2C(j+1, i, *dev_n)]);
}


__global__ void reduceBlock(const double *A, const double *Anew, const int n, double *out){
    // создание блока
    typedef cub::BlockReduce<double, 256> BlockReduce; 
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double error = 0;
    // проходим по массивам и находим макс разницу
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x){
        error = MAX(error, ABS(Anew[i] - A[i]));
    }
    // засовываем максимальную разницу в блок редукции
    double block_max_diff = BlockReduce(temp_storage).Reduce(error, cub::Max());

    // обновление значения
    if (threadIdx.x == 0){
        out[blockIdx.x] = block_max_diff; 
    }
}



//основной цикл программы
void solution(const TYPE tol, const int iter_max, const int n)
{
    //текущая ошибка, счетчик итераций, размер(площадь) сетки
    TYPE error {1.0};
    int iter{0},size{n*n}; 
    
    //матрицы
    TYPE *A = new TYPE [size], *Anew = new TYPE [size];
    
    //инициализация сеток
    initArr(A, n);
    initArr(Anew, n);

    bool flag {true}; // флаг для обновления значения ошибки на хосте

    //printArr(A,n);
    //std::cout<<"___________________________"<<std::endl;

    //размерность
    int* dev_n;
    cudaMalloc(&dev_n, sizeof(int));
    cudaMemcpy(dev_n, &n, sizeof(int), cudaMemcpyHostToDevice);

    //ошибка
    TYPE* dev_error;
    cudaMalloc(&dev_error, sizeof(int));
    cudaMemcpy(dev_error, &error, sizeof(int), cudaMemcpyHostToDevice);

    //указатели на массивы, которые будут лежать на девайсе
    double *dev_A, *dev_Anew, *dev_Atmp;

    //выделение памяти на видеокарте под массивы
    //копирование массивов на видеокарту 
    cudaMalloc(&dev_A,size*sizeof(TYPE));
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);

    cudaMalloc(&dev_Anew,size*sizeof(TYPE));
    cudaMemcpy(dev_Anew, Anew, size, cudaMemcpyHostToDevice);
    

    //определение количество потоков на блок
    dim3 threadPerBlock = dim3(32,32); 
    //определение количество блоков на сетку
    dim3 blocksPerGrid = dim3((n + 31) / 32, (n+31)/32);

    
    // количество блоков редукции
    int num_blocks_reduce {(size + 255) / 256}; 
    // выделяем память под ошибку блочной редукции
    double *error_reduction;
    cudaMalloc(&error_reduction, sizeof(double) * num_blocks_reduce);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    //cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, subs, error, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    double* tmp_err = (double*)malloc(sizeof(double)); 



    while (error > tol && iter < iter_max)
    {
        flag = !(iter % n);

        // меняем местами, чтобы не делать swap с доп переменной. работает быстрее
        Step<<<blocksPerGrid, threadPerBlock>>>(dev_A, dev_Anew, dev_n); 
        Step<<<blocksPerGrid, threadPerBlock>>>(dev_Anew, dev_A, dev_n); 

        //if(flag){
            // поблочно проходим редукцией
            reduceBlock<<<num_blocks_reduce, 255>>>(dev_A, dev_Anew, size, error_reduction); 
            // проходим редукцией по всем блокам
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, dev_error, num_blocks_reduce); 
            ////обновление ошибки на хосте 
            cudaMemcpy(tmp_err, dev_error, sizeof(double), cudaMemcpyDeviceToHost);      
        //}
        ++iter;   
    }


    std::cout << "Iterations: " << iter << std::endl<< "Error: " << error << std::endl;
    
    cudaFree(dev_A);
    cudaFree(dev_Anew);
    cudaFree(dev_n);
    cudaFree(dev_error);
    cudaFree(error_reduction);
    cudaFree(d_temp_storage);

    delete[] A;
    delete[] Anew;
}

int main(int argc, char *argv[])
{
    TYPE tol{1e-6};
    int iter_max{1000000}, n{128}; // значения для отладки, по умолчанию инициализировать нулями

    //парсинг командной строки
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

    auto start = std::chrono::high_resolution_clock::now();
    solution(tol,iter_max,n);
    auto end = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end).count();
    std::cout<<"Time (ms): "<<microseconds/1000<<std::endl;
}
