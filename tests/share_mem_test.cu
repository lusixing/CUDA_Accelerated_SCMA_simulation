#include"total_include.cuh"

#define grid_dim_x 2
#define grid_dim_y 3
#define grid_dim_z 4

#define block_dim_x  5
#define block_dim_y  1
#define block_dim_z  1

__global__ void kernel_share_mem_test(int* In, int*** out){
	//__shared__ int cache[block_dim_x][block_dim_y];
	__shared__ int cache[block_dim_x];
	int tid_x = threadIdx.x;
	//int tid_y = threadIdx.y;

	cache[tid_x] = blockIdx.x + blockIdx.y + blockIdx.z + threadIdx.x + In[0];
	//printf("block idx_x: %d idx_y:%d idx_z:%d cache value:%d \n", blockIdx.x, blockIdx.y, blockIdx.z,cache[0]);
	//printf("block idx_x: %d idx_y:%d idx_z:%d \n", blockIdx.x, blockIdx.y, blockIdx.z);

	__syncthreads();

	int cacheIndex = threadIdx.x;
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0){
		int temp = cache[0];
		out[blockIdx.x][blockIdx.y][blockIdx.z] = cache[0];
		//out[blockIdx.x] = temp;
	}
		

	//out[blockIdx.x];
}


void tsmain(){
	cudaDeviceReset();
	/*int grid_dim_x = 2;
	int grid_dim_y = 3;
	int grid_dim_z = 1;

	int block_dim_x = 4;
	int block_dim_y = 5;
	int block_dim_z = 1;*/

	dim3   grid_dim(grid_dim_x, grid_dim_y, grid_dim_z);
	dim3   block_dim(block_dim_x, block_dim_y, block_dim_z);

	//double* In;
	int* In;
	int*** Out;

	cudaMallocManaged(&In, sizeof(int));
	cudaMallocManaged(&Out, grid_dim_x* sizeof(int*));

	In[0] = 0;

	for (int i = 0; i < grid_dim_x; i++){
		cudaMallocManaged(&Out[i], grid_dim_y* sizeof(int*));
		for (int j = 0; j < grid_dim_y; j++){
			cudaMallocManaged(&Out[i][j], grid_dim_z* sizeof(int));
		}
	}

	for (int l= 0; l < 2; l++){
		kernel_share_mem_test << <grid_dim, block_dim >> >(In, Out);

		cudaDeviceSynchronize();

		for (int k = 0; k < grid_dim_z; k++){
			for (int i = 0; i < grid_dim_x; i++){
				for (int j = 0; j < grid_dim_y; j++){
					printf("%d ", Out[i][j][k]);
				}
				printf("\n");
			}
			printf("\n");
		}
	}
	
	cudaDeviceSynchronize();
	
	system("pause");
}