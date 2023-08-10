#pragma once
#include"total_include.cuh"

int** allocate_Bits_host(int N_b){
	int** bits = (int**)malloc(J * sizeof(int*));
	for (int j = 0; j < J; j++){
		bits[j] = (int*)malloc(N_b * sizeof(int));
	}
	return bits;
}

int** allocate_Symbols_host(int N_s){
	int** symbols = (int**)malloc(J * sizeof(int*));
	for (int j = 0; j < J; j++){
		symbols[j] = (int*)malloc(N_s * sizeof(int));
	}
	return symbols;
}

int** allocate_Bits_dev(int N_b){
	int** bits;// = (int**)malloc(J * sizeof(int*));
	cudaMallocManaged(&bits, J * sizeof(int*));
	for (int j = 0; j < J; j++){
		//bits[j] = (int*)malloc(N_b * sizeof(int));
		cudaMallocManaged(&bits[j], N_b * sizeof(int));
	}

	return bits;
}

int** allocate_Symbols_dev(int N_s){
	int** symbols;// = (int**)malloc(J * sizeof(int*));
	cudaMallocManaged(&symbols, J * sizeof(int*));
	for (int j = 0; j < J; j++){
		//symbols[j] = (int*)malloc(N_s * sizeof(int));
		cudaMallocManaged(&symbols[j], N_s * sizeof(int));
	}

	return symbols;
}

cuComplex*** allocate_x_host(int N_s){
	cuComplex*** x = (cuComplex***)malloc(J * sizeof(cuComplex**));
	for (int j = 0; j < J; j++){
		x[j] = (cuComplex**)malloc(N_s * sizeof(cuComplex*));
		for (int n = 0; n < N_s; n++){
			x[j][n] = (cuComplex*)malloc(K * sizeof(cuComplex));
		}
	}

	return x;
}


cuComplex*** allocate_x_dev(int N_s){
	cuComplex*** x;// = (cuComplex***)malloc(J * sizeof(cuComplex**));
	cudaMallocManaged(&x, J * sizeof(cuComplex**));
	for (int j = 0; j < J; j++){
		//x[j] = (cuComplex**)malloc(N_s * sizeof(cuComplex*));
		cudaMallocManaged(&x[j], N_s * sizeof(cuComplex*));
		for (int n = 0; n < N_s; n++){
			//x[j][n] = (cuComplex*)malloc(K * sizeof(cuComplex));
			cudaMallocManaged(&x[j][n], K * sizeof(cuComplex));
		}
	}

	return x;
}

cuComplex*** allocate_h(int N_s){
	cuComplex*** h;// = (cuComplex***)malloc(J * sizeof(cuComplex**));
	cudaMallocManaged(&h, J * sizeof(cuComplex**));
	for (int j = 0; j < J; j++){
		//h[j] = (cuComplex**)malloc(N_s * sizeof(cuComplex*));
		cudaMallocManaged(&h[j], N_s * sizeof(cuComplex*));
		for (int n = 0; n < N_s; n++){
			//h[j][n] = (cuComplex*)malloc(K * sizeof(cuComplex));
			cudaMallocManaged(&h[j][n], K * sizeof(cuComplex));
		}
	}

	return h;
}

cuComplex** allocate_y(int N_s){
	cuComplex** y;// = (cuComplex**)malloc(N_s * sizeof(cuComplex*));          //size(y) = [N_s][K]
	cudaMallocManaged(&y, N_s * sizeof(cuComplex*));
	for (int n = 0; n < N_s; n++){
		//y[n] = (cuComplex*)malloc(K * sizeof(cuComplex));
		cudaMallocManaged(&y[n], K * sizeof(cuComplex));
	}

	return y;
}

double**** allocate_f(){
	double**** f;// = (double****)malloc(K*sizeof(double***));
	cudaMallocManaged(&f, K*sizeof(double***));
	for (int k = 0; k < K; k++){
		cudaMallocManaged(&f[k], M*sizeof(double**));
		for (int m1 = 0; m1 < M; m1++){
			cudaMallocManaged(&f[k][m1], M*sizeof(double*));
			for (int m2 = 0; m2 < M; m2++){
				cudaMallocManaged(&f[k][m1][m2], M*sizeof(double));
			}
		}
	}
	return f;
}

double*** allocate_Igv_n(){
	double*** Igv; 
	
	cudaMallocManaged(&Igv, K*sizeof(double**));
	for (int k = 0; k < K; k++){
		cudaMallocManaged(&Igv[k], J*sizeof(double*));
		for (int j = 0; j < J; j++){
			cudaMallocManaged(&Igv[k][j], M*sizeof(double));
			for (int m = 0; m < M; m++){
				Igv[k][j][m] = 0;
			}
		}
	}

	return Igv;
}

double*** allocate_Ivg_n(double ap){
	double*** Ivg;

	cudaMallocManaged(&Ivg, K*sizeof(double**));
	for (int k = 0; k < K; k++){
		cudaMallocManaged(&Ivg[k], J*sizeof(double*));
		for (int j = 0; j < J; j++){
			cudaMallocManaged(&Ivg[k][j], M*sizeof(double));
			for (int m = 0; m < M; m++){
				Ivg[k][j][m] = log(ap);
			}
		}
	}
	return Ivg;
}

double**** allocate_Igv(int N){
	double**** Igv;
	cudaMallocManaged(&Igv,N*sizeof(double***) );
	for (int n = 0; n < N; n++){
		cudaMallocManaged(&Igv[n], K*sizeof(double**));
		for (int k = 0; k < K; k++){
			cudaMallocManaged(&Igv[n][k], J*sizeof(double*));
			for (int j = 0; j < J; j++){
				cudaMallocManaged(&Igv[n][k][j], M*sizeof(double));
				for (int m = 0; m < M; m++){
					Igv[n][k][j][m] = 0;
				}
			}
		}
	}
	return Igv;
}

double**** allocate_Ivg(int N){
	double**** Ivg;

	cudaMallocManaged(&Ivg, N*sizeof(double***));
	for (int n = 0; n < N; n++){
		cudaMallocManaged(&Ivg[n], K*sizeof(double**));
		for (int k = 0; k < K; k++){
			cudaMallocManaged(&Ivg[n][k], J*sizeof(double*));
			for (int j = 0; j < J; j++){
				cudaMallocManaged(&Ivg[n][k][j], M*sizeof(double));
				for (int m = 0; m < M; m++){
					Ivg[n][k][j][m] = log((double) 1/M);
				}
			}
		}

	}

	return Ivg;
}

float**** allocatef_Igv(int N){
	float**** Igv;
	cudaMallocManaged(&Igv, N*sizeof(float***));
	for (int n = 0; n < N; n++){
		cudaMallocManaged(&Igv[n], K*sizeof(float**));
		for (int k = 0; k < K; k++){
			cudaMallocManaged(&Igv[n][k], J*sizeof(float*));
			for (int j = 0; j < J; j++){
				cudaMallocManaged(&Igv[n][k][j], M*sizeof(float));
				for (int m = 0; m < M; m++){
					Igv[n][k][j][m] = 0;
				}
			}
		}
	}
	return Igv;
}

float**** allocatef_Ivg(int N){
	float**** Ivg;
	cudaMallocManaged(&Ivg, N*sizeof(float***));
	for (int n = 0; n < N; n++){
		cudaMallocManaged(&Ivg[n], K*sizeof(float**));
		for (int k = 0; k < K; k++){
			cudaMallocManaged(&Ivg[n][k], J*sizeof(float*));
			for (int j = 0; j < J; j++){
				cudaMallocManaged(&Ivg[n][k][j], M*sizeof(float));
				for (int m = 0; m < M; m++){
					Ivg[n][k][j][m] = log((float)1 / M);
				}
			}
		}
	}
	return Ivg;
}