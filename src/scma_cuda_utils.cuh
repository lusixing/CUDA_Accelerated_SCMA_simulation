#pragma once
#include"total_include.cuh"
#include"gaussian.cuh"

cuComplex*** get_default_codebook(){

	cuComplex CB[J][M][K] = {
		{ { cuComplex(0, 0), cuComplex(-0.1815, -0.1318), cuComplex(0, 0), cuComplex(0.7851, 0) },
		{ cuComplex(0, 0), cuComplex(-0.6351, -0.4615), cuComplex(0, 0), cuComplex(-0.2243, 0) },
		{ cuComplex(0, 0), cuComplex(0.6351, 0.4615), cuComplex(0, 0), cuComplex(0.2243, 0) },
		{ cuComplex(0, 0), cuComplex(0.1815, 0.1318), cuComplex(0, 0), cuComplex(-0.7851, 0) }
		},

		{ { cuComplex(0.7851, 0), cuComplex(0, 0), cuComplex(-0.1815, 0.1318), cuComplex(0, 0) },
		{ cuComplex(-0.2243, 0), cuComplex(0, 0), cuComplex(-0.6351, -0.4615), cuComplex(0, 0) },
		{ cuComplex(0.2243, 0), cuComplex(0, 0), cuComplex(0.6351, 0.4615), cuComplex(0, 0) },
		{ cuComplex(-0.7851, 0), cuComplex(0, 0), cuComplex(0.1815, 0.1318), cuComplex(0, 0) }
		},

		{ { cuComplex(-0.6351, 0.4615), cuComplex(0.1392, -0.1759), cuComplex(0, 0), cuComplex(0, 0) },
		{ cuComplex(0.1851, 0.4615), cuComplex(0.4873, -0.6156), cuComplex(0, 0), cuComplex(0, 0) },
		{ cuComplex(-0.1851, 0.1318), cuComplex(-0.4873, 0.6156), cuComplex(0, 0), cuComplex(0, 0) },
		{ cuComplex(0.6351, 0.4615), cuComplex(-0.1392, 0.1759), cuComplex(0, 0), cuComplex(0, 0) }
		},

		{ { cuComplex(0, 0), cuComplex(0, 0), cuComplex(0.7851, 0), cuComplex(-0.0055, 0.2242) },
		{ cuComplex(0, 0), cuComplex(0, 0), cuComplex(-0.2243, 0), cuComplex(-0.0193, -0.7848) },
		{ cuComplex(0, 0), cuComplex(0, 0), cuComplex(0.2243, 0), cuComplex(0.0193, 0.7848) },
		{ cuComplex(0, 0), cuComplex(0, 0), cuComplex(-0.7851, 0), cuComplex(0.0055, 0.2242) }
		},

		{ { cuComplex(-0.0055, -0.2242), cuComplex(0, 0), cuComplex(0, 0), cuComplex(-0.6351, 0.4615) },
		{ cuComplex(-0.0193, -0.7848), cuComplex(0, 0), cuComplex(0, 0), cuComplex(0.1851, -0.1318) },
		{ cuComplex(0.0193, 0.7848), cuComplex(0, 0), cuComplex(0, 0), cuComplex(-0.1815, 0.1318) },
		{ cuComplex(0.0055, 0.2242), cuComplex(0, 0), cuComplex(0, 0), cuComplex(0.6351, -0.4615) }
		},

		{ { cuComplex(0, 0), cuComplex(0.7851, 0), cuComplex(0.1392, -0.1759), cuComplex(0, 0) },
		{ cuComplex(0, 0), cuComplex(-0.2243, 0), cuComplex(0.4873, -0.6156), cuComplex(0, 0) },
		{ cuComplex(0, 0), cuComplex(0.2243, 0), cuComplex(-0.4873, 0.6156), cuComplex(0, 0) },
		{ cuComplex(0, 0), cuComplex(-0.7851, 0), cuComplex(-0.1392, 0.1759), cuComplex(0, 0) }
		},
	};


	cuComplex*** P;// = (cuComplex***)malloc(J * sizeof(cuComplex**));
	cudaMallocManaged(&P, J * sizeof(cuComplex**));
	for (int j = 0; j < J; j++){
		//P[j] = (cuComplex**)malloc(M * sizeof(cuComplex*));
		cudaMallocManaged(&P[j], M * sizeof(cuComplex*));
		for (int m = 0; m < M; m++){
			//P[j][m] = (cuComplex*)malloc(K * sizeof(cuComplex));
			cudaMallocManaged(&P[j][m], K * sizeof(cuComplex));
			for (int k = 0; k < K; k++){
				P[j][m][k] = CB[j][m][k];
			}
		}
	}


	return P;
}

void get_h(cuComplex*** h, int N_s){                  //size(h) =[J][N][K]
	double mu = 0;
	double sig = 1;

	std::random_device rd;
	std::default_random_engine generator;
	generator.seed(rd());
	std::normal_distribution<double> distribution(mu, sig);

	for (int j = 0; j < J; j++){
		for (int n = 0; n < N_s; n++){
			for (int k = 0; k < K; k++){
				h[j][n][k].r = distribution(generator);
				h[j][n][k].i = distribution(generator);
			}
		}
	}
}

void bit2dec(int* bits, int* symbols, int N_b, int Mb){

	if (N_b % Mb != 0){ printf("length mismatch"); system("pause"); }

	int len_s = N_b / Mb;

	for (int s = 0; s < len_s; s++){
		int temp = 0;
		int ind = 0;
		for (int t = Mb*s + 1; t >(s - 1)*Mb + 1; t--){
			if (bits[t] == 1){
				temp = temp + pow(2, ind);
			}
			ind++;
		}
		symbols[s] = temp;
		//cout << S[s];
	}

}

__device__ void dec2bit(int* bits, int* symbols, int N_s, int Mb){
	int len_b = N_s*Mb;
	//int* B = (int*)malloc(len_b*sizeof(int));

	for (int s = 0; s < N_s; s++){
		int temp = symbols[s];
		int ind = 1;
		for (int t = Mb*s + 1; t >(s - 1)*Mb + 1; t--){
			if (temp % (int)powf(2, ind) != 0){
				bits[t] = 1;
				temp -= (int)powf(2, ind - 1);
			}
			else{
				bits[t] = 0;
			}

			ind += 1;
		}

	}

}

void scma_encode(cuComplex*** x, cuComplex*** CB, int** Bits, int** Symbols, int N_s){        //size(x) =[J][N][K]
	int Mb = (int)log2(M);
	int N_b = N_s *log2(M);

	for (int j = 0; j < J; j++){
		bit2dec(Bits[j], Symbols[j], N_b, Mb);
		for (int n = 0; n < N_s; n++){
			//cout << S_j[n];
			x[j][n] = CB[j][Symbols[j][n]];
		}
		//cout << endl;
	}

}

void scma_uplink_transmisson(cuComplex** y, cuComplex*** h, cuComplex***x, int N_s, double p_noise){

	//size(y) = [N_s][K]

	std::random_device rd;
	std::default_random_engine generator;
	generator.seed(rd());
	std::normal_distribution<double> distribution(0, p_noise);

	for (int n = 0; n < N_s; n++){
		for (int k = 0; k < K; k++){
			y[n][k] = cuComplex(0, 0);
			for (int j = 0; j < J; j++){
				cuComplex noise = complex_noise(p_noise);
				y[n][k] = y[n][k] + x[j][n][k] * h[j][n][k] + noise;
			}
		}
	}

}


__host__ __device__ void Igv_Ivg_init(float**** Igv, float**** Ivg, int N_s){
	for (int n = 0; n < N_s; n++){
		for (int k = 0; k < K; k++){
			for (int j = 0; j < J; j++){
				//cudaMemset(&Igv[n][k][j], 0 ,M*sizeof(float));
				//cudaMemset(&Ivg[n][k][j], 1, M*sizeof(float));

				for (int m = 0; m < M; m++){
					Igv[n][k][j][m] = 0;
					Ivg[n][k][j][m] = log((float)1 / M);
				}

			}
		}
	}
}

void Igv_Ivg_test(double*** II){
	for (int k = 0; k < K; k++){
		for (int j = 0; j < J; j++){
			for (int m = 0; m < M; m++){
				cout << II[k][j][m];
			}
			cout << endl;
		}
		cout << endl;
	}
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

void memcpy_2d(int** dst, int**src, int N_rows,int N_cols){
	for (int r = 0; r < N_rows; r++){
		//for (int c = 0; c < N_cols; c++){
		//	dst[r][c] = src[r][c];
		//}
		cudaMemcpy(dst[r],src[r], N_cols*sizeof(int) , cudaMemcpyDeviceToHost);
	}



}