#include"total_include.cuh"

void CB_test(cuComplex*** CB){
	for (int j = 0; j < J; j++){
		for (int m = 0; m < M; m++){
			for (int k = 0; k < K; k++){
				printf("%f %f ", CB[j][m][k].r, CB[j][m][k].i);
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");
}

void FreeManageMemByP(double*** P, int N){
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			cudaFree(P[i][j]);
		}
		cudaFree(P[i]);
	}
	cudaFree(P);

}

void mem_managed_test(){
	int N = 150;
	double*** mem;

	size_t total_mem, free_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	std::cout << " Currently " << free_mem << " bytes free" << std::endl;

	cudaMallocManaged(&mem, N*sizeof(double**));
	for (int i = 0; i < N; i++){
		cudaMallocManaged(&mem[i], N*sizeof(double*));
		for (int j = 0; j < N; j++){
			cudaMallocManaged(&mem[i][j], N*sizeof(double));
		}

	}

	cudaMemGetInfo(&free_mem, &total_mem);
	std::cout << " Currently " << free_mem << " bytes free" << std::endl;

	printf("allocation complete\n");

	//free managed_memory way 1//
	/*for (int i = 0; i < N; i++){
	    for (int j = 0; j < N; j++){
	        cudaFree(mem[i][j]) ;
	      }
		cudaFree(mem[i]);
	}
	cudaFree(mem);*/

	//free managed_memory way 2//
	FreeManageMemByP(mem, N);

	cudaMemGetInfo(&free_mem, &total_mem);
	std::cout << " Currently " << free_mem << " bytes free" << std::endl;

	printf("free complete");
}

void MemFree_test(){
	int N = 600;
	double*** Mem = (double***)malloc(N*sizeof(double**));
	for (int i = 0; i < N; i++){
		Mem[i] = (double**)malloc(N*sizeof(double*));
		for (int j = 0; j < N; j++){
			Mem[i][j] = (double*)malloc(N*sizeof(double));
			/*for (int k = 0; k < N; k++){
			Mem[i][j][k] = (i % (N / 100))*(j % (N / 100))*(k % (N / 100));
			}*/
		}

	}

	cout << "allocation complete" << endl;

	//free memory way 1//
	/*for (int i = 0; i < N; i++){
	for (int j = 0; j < N; j++){
	free(Mem[i][j]) ;
	}
	free(Mem[i]);
	}
	free(Mem);*/

	//free memory way 2//
	/*for (int i = 0; i < N; i++){
	for (int j = 0; j < N; j++){
	delete(Mem[i][j]);
	}
	delete(Mem[i]);
	}
	delete(Mem);
	*/
	//free memory way3

	//free_variable_test(Mem, N);

	cout << "free complete" << endl;
}

void bias_test(){
	//int A[4][4][4];
	int*** A = (int***)malloc(4 * sizeof(int**));

	for (int i = 0; i < 4; i++){
		A[i] = (int**)malloc(4 * sizeof(int*));
		for (int j = 0; j < 4; j++){
			A[i][j] = (int*)malloc(4 * sizeof(int));
			for (int k = 0; k < 4; k++){
				A[i][j][k] = i + j + k;
			}
		}
	}
	printf("A:\n");
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			for (int k = 0; k < 4; k++){
				printf("%d ", A[i][j][k]);
			}
			printf("\n");
		}
		printf("\n");
	}

	int*** B = A + 2;
	printf("A:\n");
	for (int i = 0; i < 2; i++){
		for (int j = 0; j < 4; j++){
			for (int k = 0; k < 4; k++){
				printf("%d ",B[i][j][k]);
			}
			printf("\n");
		}
		printf("\n");
	}

}

void bits_test(int** bits,int N_b ){
	for (int j = 0; j < J; j++){
		for (int n = 0; n < N_b; n++){
			printf("%d", bits[j][n]);
		}
		printf("\n");
	}
	printf("\n");
}