#include"total_include.cuh"
#include"scma_cuda_utils.cuh"
//#include"scma_test.h"
#include"free_variables.cuh"

double log_sum_exp(double* x, int len){

	double y = 0;

	for (int i = 0; i < len; i++){
		y += exp(x[i]);
	}

	return  log(y);
}


int** get_J_indices(cuComplex*** CB, int deg_j){
	int** indices;// = (int**)malloc(J*sizeof(int*));
	cudaMallocManaged(&indices, J*sizeof(int*));
	for (int j = 0; j < J; j++){
		//indices[j] = (int*)malloc(deg_j * sizeof(int));
		cudaMallocManaged(&indices[j], deg_j * sizeof(int));
		int idx = 0;
		for (int k = 0; k < K; k++){
			if (CB[j][0][k].r != 0 || CB[j][0][k].i != 0){
				indices[j][idx] = k;
				idx += 1;
			}
		}
	}

	return indices;
}

int** get_K_indices(cuComplex*** CB, int deg_k){
	int** indices;// = (int**)malloc(K*sizeof(int*));
	cudaMallocManaged(&indices, K*sizeof(int*));
	for (int k = 0; k < K; k++){
		//indices[k] = (int*)malloc(deg_k * sizeof(int));
		cudaMallocManaged(&indices[k], deg_k * sizeof(int));
		int idx = 0;
		for (int j = 0; j < J; j++){
			if (CB[j][0][k].r != 0 || CB[j][0][k].i != 0){
				indices[k][idx] = j;
				idx += 1;
			}
		}
	}
	return indices;
}

void get_f_l3_log(double*** f_k, cuComplex y_k, cuComplex*** h, cuComplex*** CB, int* ind_k, int k, int n, double N0){        //size (CB) = [J][M][K]
	////size(h) =[J][N][K]          
	for (int m1 = 0; m1 < M; m1++){
		for (int m2 = 0; m2 < M; m2++){
			for (int m3 = 0; m3 < M; m3++){
				cuComplex diff = y_k - (CB[ind_k[0]][m1][k] * h[ind_k[0]][n][k] + CB[ind_k[1]][m2][k] * h[ind_k[1]][n][k] + CB[ind_k[2]][m3][k] * h[ind_k[2]][n][k]);
				f_k[m1][m2][m3] = -(1 / N0) * diff.abs() * diff.abs();
			}

		}
	}

}

void Igv_update_l3_log(double*** f, int* ind_k, double*** Igv, double*** Ivg, int k){

	for (int m1 = 0; m1 < M; m1++){
		double sIgv[M*M] = { 0 };
		int idx = 0;
		for (int m2 = 0; m2 < M; m2++){
			for (int m3 = 0; m3 < M; m3++){
				sIgv[idx] = f[m1][m2][m3] + Ivg[k][ind_k[1]][m2] + Ivg[k][ind_k[2]][m3];

				idx++;
			}
		}
		double lsp = log_sum_exp(sIgv, M*M);
		Igv[k][ind_k[0]][m1] = lsp;

		//cout << lsp<<endl;
	}

	for (int m2 = 0; m2 < M; m2++){
		double sIgv[M*M] = { 0 };
		int idx = 0;
		for (int m1 = 0; m1 < M; m1++){
			for (int m3 = 0; m3 < M; m3++){
				sIgv[idx] = f[m1][m2][m3] + Ivg[k][ind_k[0]][m1] + Ivg[k][ind_k[2]][m3];
				idx++;
			}
		}

		double lsp = log_sum_exp(sIgv, M*M);
		Igv[k][ind_k[1]][m2] = lsp;

		//cout << lsp;
	}

	for (int m3 = 0; m3 < M; m3++){
		double sIgv[M*M] = { 0 };
		int idx = 0;
		for (int m1 = 0; m1 < M; m1++){
			for (int m2 = 0; m2 < M; m2++){
				sIgv[idx] = f[m1][m2][m3] + Ivg[k][ind_k[0]][m1] + Ivg[k][ind_k[1]][m2];
				idx++;
			}
		}

		double lsp = log_sum_exp(sIgv, M*M);
		Igv[k][ind_k[2]][m3] = lsp;

		//cout << lsp;
	}
}


void Ivg_update_l2_log(int* ind_j, double*** Igv, double*** Ivg, int j){
	double lsp1 = log_sum_exp(Igv[ind_j[1]][j], M);
	double lsp2 = log_sum_exp(Igv[ind_j[0]][j], M);

	for (int m = 0; m < M; m++){
		Ivg[ind_j[0]][j][m] = Igv[ind_j[1]][j][m] - lsp1;
		Ivg[ind_j[1]][j][m] = Igv[ind_j[0]][j][m] - lsp2;
	}

}

void scma_decode_main(int** b_dec, cuComplex** y, cuComplex*** h, cuComplex*** CB, int N_s, double p_noise){
	//int** b_dec = (int**)malloc(J * sizeof(int*));
	int** s_dec;// = (int**)malloc(J * sizeof(int*));
	cudaMallocManaged(&s_dec, J * sizeof(int*));
	for (int j = 0; j < J; j++){
		//b_dec[j] = (int*)malloc(N_s*log2(M)*sizeof(int));
		//s_dec[j] = (int*)malloc(N_s *sizeof(int));
		cudaMallocManaged(&s_dec[j], N_s *sizeof(int));
	}

	int Nit = 5;
	int deg_j = 2;
	int deg_k = 3;

	double ap = 1 / M;
	double EbN0 = 1 / (CB[0][0][1].abs()*CB[0][0][1].abs()* p_noise);
	double N0 = 1 / EbN0;

	double*** Igv; //= (double***)malloc(K*sizeof(double**));   //size(Ivg) = size(Igv)= (K,J,M)
	double*** Ivg; //= (double***)malloc(K*sizeof(double**));
	cudaMallocManaged(&Igv, K*sizeof(double**));
	cudaMallocManaged(&Ivg, K*sizeof(double**));

	for (int k = 0; k < K; k++){
		//Igv[k] = (double**)malloc(J*sizeof(double*));
		//Ivg[k] = (double**)malloc(J*sizeof(double*));
		cudaMallocManaged(&Igv[k], J*sizeof(double*));
		cudaMallocManaged(&Ivg[k], J*sizeof(double*));
		for (int j = 0; j < J; j++){
			//Igv[k][j] = (double*)malloc(M*sizeof(double));
			//Ivg[k][j] = (double*)malloc(M*sizeof(double));
			cudaMallocManaged(&Igv[k][j], M*sizeof(double));
			cudaMallocManaged(&Ivg[k][j], M*sizeof(double));

			for (int m = 0; m < M; m++){
				Igv[k][j][m] = 0;
				Ivg[k][j][m] = log(ap);
			}
		}
	}

	double**** f;// = (double****)malloc(K*sizeof(double***));
	cudaMallocManaged(&f, K*sizeof(double***));
	for (int k = 0; k < K; k++){
		//f[k] = (double***)malloc(M*sizeof(double**));
		cudaMallocManaged(&f[k], M*sizeof(double**));
		for (int m1 = 0; m1 < M; m1++){
			//f[k][m1] = (double**)malloc(M*sizeof(double*));
			cudaMallocManaged(&f[k][m1], M*sizeof(double*));
			for (int m2 = 0; m2 < M; m2++){
				//f[k][m1][m2] = (double*)malloc(M*sizeof(double));
				cudaMallocManaged(&f[k][m1][m2], M*sizeof(double));
			}
		}
	}


	int** Ind_J = get_J_indices(CB, deg_j);
	int** Ind_K = get_K_indices(CB, deg_k);

	for (int n = 0; n < N_s; n++){

		Igv_Ivg_init(Igv, Ivg);

		for (int k = 0; k < K; k++){
			get_f_l3_log(f[k], y[n][k], h, CB, Ind_K[k], k, n, N0);         //size(y) = [N_s][K]
			//f_test(f[k]);
		}

		for (int iter = 0; iter < Nit; iter++){
			for (int k = 0; k < K; k++){
				Igv_update_l3_log(f[k], Ind_K[k], Igv, Ivg, k);
			}

			for (int j = 0; j < J; j++){
				Ivg_update_l2_log(Ind_J[j], Igv, Ivg, j);
			}

		}

		//final decision
		for (int j = 0; j < J; j++){
			double Q_max = -99999;
			//int max_ind = 0;
			for (int m = 0; m < M; m++){

				double Q_m = 0;
				for (int i = 0; i < deg_j; i++){
					Q_m += Igv[Ind_J[j][i]][j][m];

					//cout << Igv[Ind_J[j][i]][j][m];
				}
				//cout << Q_m;
				if (Q_m > Q_max){
					Q_max = Q_m;
					s_dec[j][n] = m;
				}
			}
		}

	}


	for (int j = 0; j < J; j++){
		dec2bit( b_dec[j], s_dec[j], N_s, 2);
	}

	free_variables11(Ind_K, Ind_J, Igv, Ivg, f, s_dec);

	//return b_dec;

	//size_t total_mem, free_mem;
	//cudaMemGetInfo(&free_mem, &total_mem);
	//std::cout << " Currently " << free_mem << " bytes free" << std::endl;
}