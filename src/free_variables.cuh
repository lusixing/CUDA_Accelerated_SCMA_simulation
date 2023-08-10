#pragma once
#include"total_include.cuh"

void free_variable_test(double*** Mem, int N){
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			free(Mem[i][j]);
		}
		free(Mem[i]);
	}
	free(Mem);
}

void free_variables1(double*** Igv, double*** Ivg, double**** f, int** s_dec){
	for (int k = 0; k < K; k++){
		for (int j = 0; j < J; j++){
			free(Igv[k][j]);
			free(Ivg[k][j]);
		}
		free(Igv[k]);
		free(Ivg[k]);
	}
	free(Igv);
	free(Ivg);

	for (int k = 0; k < K; k++){
		for (int m1 = 0; m1 < M; m1++){
			for (int m2 = 0; m2 < M; m2++){
				free(f[k][m1][m2]);
			}
			free(f[k][m1]);
		}
		free(f[k]);
	}
	free(f);


	for (int j = 0; j < J; j++){
		free(s_dec[j]);
	}
	free(s_dec);
}

void free_variables11(int** indices_K, int** indices_J, double*** Igv_n, double*** Ivg_n, double**** f, int** s_dec){
	for (int k = 0; k < K; k++){
		cudaFree(indices_K[k]);
	}
	cudaFree(indices_K);

	for (int j = 0; j < J; j++){
		cudaFree(indices_J[j]);
	}
	cudaFree(indices_J);

	for (int k = 0; k < K; k++){
		for (int j = 0; j < J; j++){
			cudaFree(Igv_n[k][j]);
			cudaFree(Ivg_n[k][j]);
		}
		cudaFree(Igv_n[k]);
		cudaFree(Ivg_n[k]);
	}
	cudaFree(Igv_n);
	cudaFree(Ivg_n);

	for (int k = 0; k < K; k++){
		for (int m1 = 0; m1 < M; m1++){
			for (int m2 = 0; m2 < M; m2++){
				cudaFree(f[k][m1][m2]);
			}
			cudaFree(f[k][m1]);
		}
		cudaFree(f[k]);
	}
	cudaFree(f);


	for (int j = 0; j < J; j++){
		cudaFree(s_dec[j]);
	}
	cudaFree(s_dec);
}

void free_variables12(int** indices_K, int** indices_J, double**** Igv, double**** Ivg, int** s_dec, int N_s){
	for (int k = 0; k < K; k++){
		cudaFree(indices_K[k]);
	}
	cudaFree(indices_K);

	for (int j = 0; j < J; j++){
		cudaFree(indices_J[j]);
	}
	cudaFree(indices_J);

	for (int n = 0; n < N_s; n++){
		for (int k = 0; k < K; k++){
			for (int j = 0; j < J; j++){
				cudaFree(Igv[n][k][j]);
				cudaFree(Ivg[n][k][j]);
			}
			cudaFree(Igv[n][k]);
			cudaFree(Ivg[n][k]);
		}
		cudaFree(Igv[n]);
		cudaFree(Ivg[n]);
	}
	cudaFree(Igv);
	cudaFree(Ivg);

	for (int j = 0; j < J; j++){
		cudaFree(s_dec[j]);
	}
	cudaFree(s_dec);
}

void free_variables12(int** indices_K, int** indices_J, float**** Igv, float**** Ivg, int** s_dec, int N_s){
	for (int k = 0; k < K; k++){
		cudaFree(indices_K[k]);
	}
	cudaFree(indices_K);

	for (int j = 0; j < J; j++){
		cudaFree(indices_J[j]);
	}
	cudaFree(indices_J);

	for (int n = 0; n < N_s; n++){
		for (int k = 0; k < K; k++){
			for (int j = 0; j < J; j++){
				cudaFree(Igv[n][k][j]);
				cudaFree(Ivg[n][k][j]);
			}
			cudaFree(Igv[n][k]);
			cudaFree(Ivg[n][k]);
		}
		cudaFree(Igv[n]);
		cudaFree(Ivg[n]);
	}
	cudaFree(Igv);
	cudaFree(Ivg);

	for (int j = 0; j < J; j++){
		cudaFree(s_dec[j]);
	}
	cudaFree(s_dec);
}


void free_variables2(cuComplex*** x, cuComplex*** h, cuComplex** y, int N_s){

	for (int j = 0; j < J; j++){

		for (int n = 0; n < N_s; n++){
			free(h[j][n]);
		}
		free(h[j]);
	}
	free(h);

	for (int j = 0; j < J; j++){
		//for (int n = 0; n < N_s; n++){
		//	free(x[j][n]);
		//}
		free(x[j]);
	}
	free(x);

	for (int n = 0; n < N_s; n++){
		free(y[n]);
	}
	free(y);

}

void free_variables22(cuComplex*** x, cuComplex*** h, cuComplex** y, int N_s){

	for (int j = 0; j < J; j++){

		for (int n = 0; n < N_s; n++){
			cudaFree(h[j][n]);
		}
		cudaFree(h[j]);
	}
	cudaFree(h);

	for (int j = 0; j < J; j++){
		//for (int n = 0; n < N_s; n++){
		//	free(x[j][n]);
		//}
		cudaFree(x[j]);
	}
	cudaFree(x);

	for (int n = 0; n < N_s; n++){
		cudaFree(y[n]);
	}
	cudaFree(y);

}