#include <cstdio>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define N 70000
#define M 1000

class ObjBox
{
public:

	int oid;
	float x;
	float y;
	float ts;
};

class Bucket
{
public:

	int bid;
	int nxt;
	ObjBox *arr_obj;
	int nO;
};

int main3()
{

	Bucket *arr_bkt;
	cudaMallocManaged(&arr_bkt, N * sizeof(Bucket));

	for (int i = 0; i < N; i++) {
		arr_bkt[i].bid = i;
		arr_bkt[i].nxt = -1;
		arr_bkt[i].nO = 0;

		size_t allocsz = size_t(M) * sizeof(ObjBox);
		cudaError_t r = cudaMallocManaged(&(arr_bkt[i].arr_obj), allocsz);
		if (r != cudaSuccess) {
			printf("CUDA Error on %s\n", cudaGetErrorString(r));
			exit(0);
		}
		else {
			size_t total_mem, free_mem;
			cudaMemGetInfo(&free_mem, &total_mem);
			std::cout << i << ":Allocated " << allocsz;
			std::cout << " Currently " << free_mem << " bytes free" << std::endl;
		}

		for (int j = 0; j < M; j++) {
			arr_bkt[i].arr_obj[j].oid = -1;
			arr_bkt[i].arr_obj[j].x = -1;
			arr_bkt[i].arr_obj[j].y = -1;
			arr_bkt[i].arr_obj[j].ts = -1;
		}
	}

	std::cout << "Bucket Array Initial Completed..." << std::endl;
	cudaFree(arr_bkt);

	return 0;
}