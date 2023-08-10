#include"stdio.h"
#include"stdlib.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void kernel_fill(int** array_dev,int N_rows, int N_cols ){
	for (int i = 0; i < N_rows; i++){
		for (int j = 0; j < N_cols; j++){
			array_dev[i][j] = i+j;
		}
	}
}

__global__ void kernel_output(int** array_dev, int N_rows, int N_cols){
	for (int i = 0; i < N_rows; i++){
		for (int j = 0; j < N_cols; j++){
			printf("%d ", array_dev[i][j]);
		}
		printf("\n");
	}
}

void tmmain(){
	int N_rows = 5;
	int N_cols = 6;

	int* a_host;
	int** b_dev;

	a_host =(int*)malloc(N_cols*sizeof(int));
	for (int i = 0; i < N_rows; i++){
		a_host[i] = i;
		
	}

	cudaMalloc((void**)&b_dev, N_rows*sizeof(int*));
	for (int i = 0; i < N_rows; i++){
		cudaMalloc((void**)&b_dev[i], N_cols*sizeof(int));
	}

	

	//cudaMemcpy(a_dev, a_host, N * sizeof(int), cudaMemcpyHostToDevice);

	kernel_fill << <1, 1 >> >(b_dev, N_rows,N_cols);
	cudaDeviceSynchronize();
	kernel_output << <1, 1 >> >(b_dev, N_rows, N_cols);
	cudaDeviceSynchronize();
	system("pause");
}