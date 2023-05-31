#include <iostream>
#include <cmath>
#include <chrono>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/cub.cuh>


// индексация
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))


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


//Фукнция для вычисления среднего значения пятиточечным шаблоном по соседним элементам
//для граничных значений сетки
__global__ void calcAverageBounds(const TYPE *A, TYPE *Anew, const int n, const int sizePerGpu)
{
    // вычисление ячейки
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    // проверка границ
	if(i <= n - 2 && i > 0){
        // среднее по соседним элементам
		Anew[1 * n + i] = 0.25 * (A[1 * n + i - 1] + A[(1 - 1) * n + i] + A[(1 + 1) * n + i] + A[1 * n + i + 1]);
		Anew[(sizePerGpu - 2) * n + j] = 0.25 * (A[(sizePerGpu - 2) * n + j - 1] + A[((sizePerGpu - 2) - 1) * n + j] + A[((sizePerGpu - 2) + 1) * n + j] + A[(sizePerGpu - 2) * n + j + 1]);
	}
}

//фукнция для вычисления среднего значения пятиточечным шаблоном по соседним элементам 
//для внутренних значений сетки
__global__ void calcAverageInnards(const TYPE *A, TYPE *Anew,const int n, const int sizePerGpu)
{
    // вычисление ячейки
	int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // проверка границ
	if(j >= 1 && i >= 2 && j <= n - 2 && i <= sizePerGpu - 2){
       // среднее по соседним элементам
		Anew[i*n + j] = 0.25 * (A[i * n + j - 1] + A[i * n + j + 1] + A[(i - 1) * n + j] + A[(i + 1) * n + j]);
	}
}

//фукнция, вычисляющая разность двух матриц
__global__ void calcMatrDiff(const TYPE *A, const TYPE *Anew, TYPE *Atmp,const int n,const int sizePerGpu){
    // вычисление ячейки
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

    // проверка границ
	if(j > 0 && i > 0 && j < n - 1 && i < sizePerGpu - 1){
        // среднее по соседним элементам
		Atmp[i * n + j] = ABS(Anew[i * n + j] - A[i * n + j]);
	}
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


// основной цикл программы
void solution(const TYPE tol, const int iter_max, const int n, const int rank, const int nRanks)
{
    // выбор гпу
	cudaSetDevice(rank);

    // текущая ошибка, счетчик итераций, размер(площадь) сетки
    TYPE error {1.0};
    int iter{0}, size{n * n};

    bool flag{true}; // флаг для обновления значения ошибки на хосте

    // матрицы
	TYPE *A;
    cudaMallocHost(&A, size * sizeof(TYPE));

	size_t sizeForOne {n / nRanks}, beginIdx {n / nRanks * rank};
    
    // инициализация сеток
    initArr(A,n);


	if(nRanks!=1){
		if (rank != 0 && rank != nRanks - 1) 
            sizeForOne += 2;
		else 
            sizeForOne += 1;   
	 }


    // указатели на массивы, которые будут лежать на девайсе
    TYPE *dev_A, *dev_Anew, *dev_Atmp;

    // выделение памяти на видеокарте под массивы
    // копирование массивов на видеокарту
	cudaMalloc((void**)&dev_A, n * sizeForOne * sizeof(TYPE));
	cudaMalloc((void**)&dev_Anew, n * sizeForOne * sizeof(TYPE));
	cudaMalloc((void**)&dev_Atmp, n * sizeForOne * sizeof(TYPE));

	size_t offset = (rank != 0) ? n : 0;
 	cudaMemcpy(dev_A, (A + (beginIdx * n) - offset), sizeof(TYPE) * n * sizeForOne, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Anew, dev_A + (beginIdx * n) - offset, sizeof(TYPE) * n * sizeForOne, cudaMemcpyDeviceToDevice);

	// создаем потоки и назначаем приоритет
	int leastPriority {0}, greatestPriority {0};
    // Получаем диапазон приоритетов потоков устройства CUDA
	cudaStream_t stream_boundaries, stream_inner;
	cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    // Создаем поток stream_boundaries с наивысшим приоритетом
	cudaStreamCreateWithPriority(&stream_boundaries, cudaStreamDefault, greatestPriority);
    // Создаем поток stream_inner с наименьшим приоритетом
	cudaStreamCreateWithPriority(&stream_inner, cudaStreamDefault, leastPriority);

    // определение количества потоков на блок
    int threads_in_block{32};
    dim3 threadPerBlock = dim3(threads_in_block, threads_in_block);

    // определение количества блоков на сетку
    dim3 blocksPerGrid(n / ((n<threads_in_block*threads_in_block)? n:threads_in_block*threads_in_block), sizeForOne);

    //ошибка
	TYPE* dev_error;
    cudaMalloc(&dev_error, sizeof(TYPE));

	// определяем требования к временному хранилищу устройства и выделяем память
    TYPE *d_temp_storage = nullptr;
    uint64_t temp_storage_bytes {0};
    // Первый вызов, чтобы предоставить количество байтов, необходимое для временного хранения, необходимого CUB.
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dev_Atmp, dev_error, size);
    // Выделение памяти под буфер
    cudaMalloc(&d_temp_storage, temp_storage_bytes);


	while(error > tol && iter < iter_max){

        flag = !(iter % n);

		// расчет границ
		calcAverageBounds<<<n, 1, 0, stream_boundaries>>>(dev_A, dev_Anew, n, sizeForOne);
		// расчет внутренних значений
		calcAverageInnards<<<threadPerBlock, blocksPerGrid, 0, stream_inner>>>(dev_A, dev_Anew, n, sizeForOne);
        //синхронизация
        cudaStreamSynchronize(stream_boundaries);


		if (flag){
            //вычитание матриц
			calcMatrDiff<<<threadPerBlock, blocksPerGrid, 0, stream_inner>>>(dev_A, dev_Anew, dev_Atmp, n, sizeForOne);
			// Результат сохраняется в dev_error, выделенной памяти d_temp_storage, и размере 
             //редукция
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dev_Atmp, dev_error, n * sizeForOne, stream_inner);
			// Синхронизируем поток stream_inner, чтобы убедиться, что все операции завершены
            cudaStreamSynchronize(stream_inner);
            // Выполняем операцию MPI_Allreduce для получения максимального значения ошибки
			MPI_Allreduce((void*)dev_error, (void*)dev_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            //обновление ошибки на хосте
			cudaMemcpyAsync(&error, dev_error, sizeof(TYPE), cudaMemcpyDeviceToHost, stream_inner);
		}
		
		// верхняя граница
        // Проверяем, если текущий процесс не является первым (rank != 0), то отправляем верхнюю границу массива dev_Anew 
        // на предыдущий процесс и одновременно принимаем верхнюю границу от предыдущего процесса.
        if (rank != 0){
		    MPI_Sendrecv(dev_Anew + n + 1, n - 2, MPI_DOUBLE, rank - 1, 0, 
				dev_Anew + 1, n - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		// нижняя граница
        // Проверяем, если текущий процесс не является последним (rank != nRanks - 1), то отправляем нижнюю границу массива dev_Anew 
        // на следующий процесс и одновременно принимаем нижнюю границу от следующего процесса.
		if (rank != nRanks - 1){
		    MPI_Sendrecv(dev_Anew + (sizeForOne - 2) * n + 1, n - 2, MPI_DOUBLE, rank + 1, 0,
					dev_Anew + (sizeForOne - 1) * n + 1, 
					n - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
        // Синхронизируем поток stream_inner, чтобы убедиться, что все операции завершены
		cudaStreamSynchronize(stream_inner);

		// swap
		TYPE* tmp = dev_A;
		dev_A = dev_Anew;
		dev_Anew = tmp;

        iter++;
	}

    
    if (rank==0) {
        std::cout << "Iterations: " << iter << std::endl<< "Error: " << error << std::endl;
    }


    //очистка выделенное памяти
    cudaFree(A);
    cudaFree(dev_A);
    cudaFree(dev_Anew);
    cudaFree(dev_Atmp);
    cudaFree(dev_error);

    cudaStreamDestroy(stream_inner)
    cudaStreamDestroy(stream_boundaries)
}


int main(int argc, char* argv[]){

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

    // Объявление и инициализация переменных rank и nRanks для определения ранга и общего числа процессов в MPI.
    int rank, nRanks;
    // Инициализация MPI.
    MPI_Init(&argc, &argv);
    // Получение ранга текущего процесса.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Получение общего числа процессов.
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    solution(tol, iter_max, n,rank,nRanks);
    
    MPI_Finalize();

    //подсчёт времени работы программы
    auto end = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end).count();

    if (rank==0) {
        std::cout << "Time (ms): " << microseconds / 1000 << std::endl;
    }

	return 0;
}