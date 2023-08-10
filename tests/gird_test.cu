#include"total_include.cuh"


__global__ void kernel_test(int* count){


	//printf(" number of dimensions of  blocks  x:%d  y:%d  z: %d\n", gridDim.x, gridDim.y, gridDim.z);

	//printf(" launching  block of idx x:%d  idx y:%d  idx z:%d\n", blockIdx.x, blockIdx.y, blockIdx.z);
	
	//printf("dimension of a block %d%d%d\n", blockDim.x, blockDim.y, blockDim.z);

	printf(" launching block of idx x:%d  idx y:%d  idx z:%d ,thread of idx x:%d  idx y:%d  idx z:%d \n ", blockIdx.x, blockIdx.y, blockIdx.z,threadIdx.x,threadIdx.y,threadIdx.z);

	//__syncthreads();

	atomicAdd(&count[0],1 );

	__syncthreads();
}


void t4main(){
	cudaDeviceReset();

	int* N;
	cudaMallocManaged(&N,sizeof(int));

	//printf("%d", N[0]);

	//printf("%d", N[1]);

	int grid_dim_x = 2;
	int grid_dim_y = 3;
	int grid_dim_z = 4;

	int block_dim_x = 5;
	int block_dim_y = 6;
	int block_dim_z = 7;

	dim3   grid_dim(grid_dim_x, grid_dim_y, grid_dim_z);
	dim3   block_dim(block_dim_x, block_dim_y, block_dim_z);

	kernel_test << <grid_dim, block_dim >> >(N);

	cudaDeviceSynchronize();

	printf("\n");
	printf("%d", N[0]);
	system("pause");
}

