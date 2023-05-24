#include <iostream>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/cub.cuh>

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

// индексация по фортрану
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

// Макрос проверки статуса операции CUDA
#define CUDA_CHECK(err)                                                        \
    {                                                                          \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess)                                               \
        {                                                                      \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                            \
        }                                                                      \
    }

// функция инициализации сетки
void initArr(TYPE *A, const int n)
{
    // заполнение углов сетки
    A[IDX2C(0, 0, n)] = 10.0;
    A[IDX2C(0, n - 1, n)] = 20.0;
    A[IDX2C(n - 1, 0, n)] = 20.0;
    A[IDX2C(n - 1, n - 1, n)] = 30.0;

    // заполнение краёв сетки
    for (int i{1}; i < n - 1; ++i)
    {
        A[IDX2C(0, i, n)] = 10 + (i * 10.0 / (n - 1));
        A[IDX2C(i, 0, n)] = 10 + (i * 10.0 / (n - 1));
        A[IDX2C(n - 1, i, n)] = 20 + (i * 10.0 / (n - 1));
        A[IDX2C(i, n - 1, n)] = 20 + (i * 10.0 / (n - 1));
    }
}

// функция печати массива
void printArr(TYPE *A, const int n)
{
    for (int i{0}; i < n; ++i)
    {
        for (int j{0}; j < n; ++j)
        {

            printf("%lf ", A[IDX2C(i, j, n)]);
        }
        std::cout << std::endl;
    }
}

// Шаг алгоритма
//фукнция для вычисления среднего значения пятиточечным шаблоном по соседним элементам
__global__ void calcAverage(const TYPE *A, TYPE *Anew, const int dev_n)
{
    // вычисление ячейки
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    // проверка границ
    if (j == 0 || i == 0 || i >= dev_n - 1 || j >= dev_n - 1) return;
    // среднее по соседним элементам
    Anew[IDX2C(j, i, dev_n)] = 0.25 * (A[IDX2C(j, i + 1, dev_n)] + A[IDX2C(j, i - 1, dev_n)] + A[IDX2C(j - 1, i, dev_n)] + A[IDX2C(j + 1, i, dev_n)]);
}


//фукнция, вычисляющая разность двух матриц
__global__ void calcMatrDiff(const TYPE *A, const TYPE *Anew, TYPE *Atmp,const int dev_n)
{
    // вычисление ячейки
    unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int i = blockDim.y * blockIdx.y + threadIdx.y;

    // проверка границ
    if (j == 0 || i == 0 || i >= dev_n - 1 || j >= dev_n - 1) return;

    //вычисление индекса
    uint32_t idx = IDX2C(j, i, dev_n);

    //результат разности двух матриц
    Atmp[idx] = ABS(Anew[idx] - A[idx]);
}

// основной цикл программы
void solution(const TYPE tol, const int iter_max, const int n)
{
    // текущая ошибка, счетчик итераций, размер(площадь) сетки
    TYPE *error;
    CUDA_CHECK(cudaMallocHost(&error, sizeof(TYPE)));
    *error = 1.0;
    int iter{0}, size{n * n};

    // матрицы
    TYPE *A;
    CUDA_CHECK(cudaMallocHost(&A, size * sizeof(TYPE)));

    // инициализация сеток
    initArr(A, n);
    //initArr(Anew, n);
    bool flag{true}; // флаг для обновления значения ошибки на хосте

    // printArr(A,n);
    // std::cout<<"___________________________"<<std::endl;

    cudaSetDevice(0);

    // определение количества потоков на блок
    int threads_in_block{32};
    dim3 threadPerBlock = dim3(threads_in_block, threads_in_block);

    // определение количества блоков на сетку
    int blocks_in_grid = ceil((TYPE)n / threads_in_block);
    dim3 blocksPerGrid = dim3(blocks_in_grid, blocks_in_grid);

    // ошибка
    TYPE *dev_error;
    CUDA_CHECK(cudaMalloc(&dev_error, sizeof(TYPE)));
    CUDA_CHECK(cudaMemcpy(dev_error, error, sizeof(TYPE), cudaMemcpyHostToDevice));

    // указатели на массивы, которые будут лежать на девайсе
    TYPE *dev_A, *dev_Anew, *dev_Atmp;

    // выделение памяти на видеокарте под массивы
    // копирование массивов на видеокарту
    CUDA_CHECK(cudaMalloc(&dev_A, size * sizeof(TYPE)));
    CUDA_CHECK(cudaMemcpy(dev_A, A, size * sizeof(TYPE), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&dev_Anew, size * sizeof(TYPE)));
    CUDA_CHECK(cudaMemcpy(dev_Anew, dev_A, size * sizeof(TYPE), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMalloc(&dev_Atmp, size * sizeof(TYPE)));

    TYPE *temp_storage = nullptr;
    uint64_t temp_storage_bytes{0};

    // Первый вызов, чтобы предоставить количество байтов, необходимое для временного хранения, необходимого CUB.
    cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, dev_Atmp, dev_error, size);

    // Выделение памяти под буфер
    CUDA_CHECK(cudaMalloc(&temp_storage, temp_storage_bytes));

    while (*error > tol && iter < iter_max)
    {

        
        flag = !(iter % n);

        // меняем местами, чтобы не делать swap с доп переменной. работает быстрее
        calcAverage<<<blocksPerGrid, threadPerBlock>>>(dev_A, dev_Anew, n);
        calcAverage<<<blocksPerGrid, threadPerBlock>>>(dev_Anew, dev_A, n);


        if (flag)
        {
            //вычитание матриц
            calcMatrDiff<<<blocksPerGrid, threadPerBlock>>>(dev_A, dev_Anew, dev_Atmp, n);
            //редукция
            cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, dev_Atmp, dev_error, size);
            //обновление ошибки на хосте
            CUDA_CHECK(cudaMemcpy(error, dev_error, sizeof(TYPE), cudaMemcpyDeviceToHost));
        }

        iter += 2;
    }

    std::cout << "Iterations: " << iter << std::endl<< "Error: " << *error << std::endl;

    //очистка выделенное памяти
    cudaFree(A);
    cudaFree(dev_A);
    cudaFree(dev_Anew);
    cudaFree(dev_Atmp);
    cudaFree(dev_error);
}

int main(int argc, char *argv[])
{
    TYPE tol{1e-6};
    int iter_max{1000000}, n{128}; // значения для отладки, по умолчанию инициализировать нулями

    // парсинг командной строки
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
    solution(tol, iter_max, n);
    auto end = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end).count();
    std::cout << "Time (ms): " << microseconds / 1000 << std::endl;
}
