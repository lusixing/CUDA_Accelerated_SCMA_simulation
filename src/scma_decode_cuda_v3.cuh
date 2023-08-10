#include"total_include.cuh"
#include"scma_cuda_utils.cuh"
//#include"scma_test.h"
#include"free_variables.cuh"
#include"space_allocate.cuh"

__device__ __host__ double log_sum_exp(double* x, int len){
	double y = 0;

	for (int i = 0; i < len; i++){
		y += exp(x[i]);
	}

	return  log(y);
}

__device__ __host__ double log_sum_exp(float* x, int len){
	double y = 0;

	for (int i = 0; i < len; i++){
		y += exp(x[i]);
	}

	return  log(y);
}


__device__ int TowD_2_linear(int* idx_2d){
	int idx_linear = M*idx_2d[0] + idx_2d[1];

	return idx_linear;
}

__device__ int* linear_2_2D(int idx_linear){
	int idx_2d[2] = { 0 };

	for (int idx1 = 1; idx1 <= M; idx1++){
		if (idx1*M > idx_linear){
			idx_2d[0] = idx1 - 1;
			idx_linear -= (idx1 - 1)*M;
			break;
		}
	}

	for (int idx2 = 1; idx2 <= M; idx2++){
		if (idx2 > idx_linear){
			idx_2d[1] = idx2 - 1;
			break;
		}
	}

	return idx_2d;
}

__device__ int* linear_2_3D(int idx_linear){
	int idx_3d[3] = { 0 };

	for (int idx1 = 1; idx1 <= M; idx1++){
		if (idx1*M*M > idx_linear){
			idx_3d[0] = idx1 - 1;
			idx_linear -= (idx1 - 1)*M*M;
			break;
		}
	}

	for (int idx2 = 1; idx2 <= M; idx2++){
		if (idx2*M > idx_linear){
			idx_3d[1] = idx2 - 1;
			idx_linear -= (idx2 - 1)*M;
			break;
		}
	}

	for (int idx3 = 1; idx3 <= M; idx3++){
		if (idx3 > idx_linear){
			idx_3d[2] = idx3 - 1;
			break;
		}
	}

	return idx_3d;
}

__global__ void Igv_update6_l3_log_kernel(cuComplex** y, cuComplex*** h, cuComplex*** CB, int** Ind_K, float**** Igv, float**** Ivg, double N0, int bias){
	int k = blockIdx.x;
	int idx_dr = blockIdx.y;
	int n = blockIdx.z + bias;

	double temp;

	int m1;
	int m2;
	int m3;
	int idx_temp;
	int* idx_2d;
	cuComplex diff(0, 0);
	double f_temp;

	__shared__ double Sigv[M*M];

	switch (idx_dr){

	case 0:
		idx_temp = threadIdx.x;
		idx_2d = linear_2_2D(idx_temp);
		m2 = idx_2d[0];
		m3 = idx_2d[1];


		//idx_temp = m2*M + m3;
		for (int m1 = 0; m1 < M; m1++){
			diff = y[n][k] - (CB[Ind_K[k][0]][m1][k] * h[Ind_K[k][0]][n][k] + CB[Ind_K[k][1]][m2][k] * h[Ind_K[k][1]][n][k] + CB[Ind_K[k][2]][m3][k] * h[Ind_K[k][2]][n][k]);
			f_temp = -(1 / N0) * diff.abs() * diff.abs();

			Sigv[idx_temp] = exp(f_temp + Ivg[n][k][Ind_K[k][1]][m2] + Ivg[n][k][Ind_K[k][2]][m3]);
			//Sigv[idx_temp] = exp(f[k][m1][m2][m3] + Ivg[k][Ind_K[k][1]][m2] + Ivg[k][Ind_K[k][2]][m3]);

			__syncthreads();
			int i = M*M / 2;
			while (i != 0) {
				if (idx_temp < i)
					Sigv[idx_temp] += Sigv[idx_temp + i];
				__syncthreads();
				i /= 2;
			}

			if (idx_temp == 0)
				Igv[n][k][Ind_K[k][0]][m1] = log(Sigv[0]);
		}


	case 1:
		idx_temp = threadIdx.x;
		idx_2d = linear_2_2D(idx_temp);
		m1 = idx_2d[0];
		m3 = idx_2d[1];

		for (int m2 = 0; m2 < M; m2++){
			diff = y[n][k] - (CB[Ind_K[k][0]][m1][k] * h[Ind_K[k][0]][n][k] + CB[Ind_K[k][1]][m2][k] * h[Ind_K[k][1]][n][k] + CB[Ind_K[k][2]][m3][k] * h[Ind_K[k][2]][n][k]);
			f_temp = -(1 / N0) * diff.abs() * diff.abs();

			Sigv[idx_temp] = exp(f_temp + Ivg[n][k][Ind_K[k][0]][m1] + Ivg[n][k][Ind_K[k][2]][m3]);

			__syncthreads();

			int i = M*M / 2;
			while (i != 0) {
				if (idx_temp < i)
					Sigv[idx_temp] += Sigv[idx_temp + i];
				__syncthreads();
				i /= 2;
			}

			if (idx_temp == 0)
				Igv[n][k][Ind_K[k][1]][m2] = log(Sigv[0]);
		}


	case 2:
		idx_temp = threadIdx.x;
		idx_2d = linear_2_2D(idx_temp);
		m1 = idx_2d[0];
		m2 = idx_2d[1];

		for (int m3 = 0; m3 < M; m3++){
			diff = y[n][k] - (CB[Ind_K[k][0]][m1][k] * h[Ind_K[k][0]][n][k] + CB[Ind_K[k][1]][m2][k] * h[Ind_K[k][1]][n][k] + CB[Ind_K[k][2]][m3][k] * h[Ind_K[k][2]][n][k]);
			f_temp = -(1 / N0) * diff.abs() * diff.abs();

			Sigv[idx_temp] = exp(f_temp + Ivg[n][k][Ind_K[k][0]][m1] + Ivg[n][k][Ind_K[k][1]][m2]);
			//Sigv[idx_temp] = exp(f[k][m1][m2][m3] + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][1]][m2]);

			__syncthreads();

			int i = M*M / 2;
			while (i != 0) {
				if (idx_temp < i)
					Sigv[idx_temp] += Sigv[idx_temp + i];
				__syncthreads();
				i /= 2;
			}

			if (idx_temp == 0)
				Igv[n][k][Ind_K[k][2]][m3] = log(Sigv[0]);
		}

	}
}

__global__ void Ivg_update_l2_log_kernel(int** Ind_J, float**** Igv, float**** Ivg, int bias){

	int n = blockIdx.x + bias;
	int j = blockIdx.y;
	
	double lsp1 = log_sum_exp(Igv[n][Ind_J[j][1]][j], M);
	double lsp2 = log_sum_exp(Igv[n][Ind_J[j][0]][j], M);

	for (int m = 0; m < M; m++){
			Ivg[n][Ind_J[j][0]][j][m] = Igv[n][Ind_J[j][1]][j][m] - lsp1;
			Ivg[n][Ind_J[j][1]][j][m] = Igv[n][Ind_J[j][0]][j][m] - lsp2;
	}
}

__global__ void final_decesion(int** Ind_J, float**** Igv, int** s_dec, int bias){

	int n = blockIdx.x + bias;
	int j = blockIdx.y;


	double Q_max = -99999;
	for (int m = 0; m < M; m++){
			double Q_m = 0;
			for (int i = 0; i < Dc; i++){
				Q_m += Igv[n][Ind_J[j][i]][j][m];
			}
			if (Q_m > Q_max){
				Q_max = Q_m;
				s_dec[j][n] = m;
			}
	}

}

__global__ void kernel_wrapup(cuComplex** y, cuComplex*** h, cuComplex*** CB, int N_partial, int** Ind_K, int** Ind_J,
	                         float**** Igv, float**** Ivg, double N0, int** s_dec , int bias){
	int Nit = 4;

	dim3   grid_dim1(K, Dr, N_partial);
	dim3   block_dim1(M*M, 1, 1);

	dim3   grid_dim2(N_partial, J, 1);
	dim3   block_dim2(1, 1, 1);

	dim3   grid_dim3(N_partial, J, 1);
	dim3   block_dim3(1, 1, 1);
	for (int iter = 0; iter < Nit; iter++){
		Igv_update6_l3_log_kernel << <grid_dim1, block_dim1 >> >(y, h, CB, Ind_K, Igv, Ivg, N0, bias);
		cudaDeviceSynchronize();

		Ivg_update_l2_log_kernel << <grid_dim2, block_dim2 >> >(Ind_J, Igv, Ivg, bias);
		cudaDeviceSynchronize();
	}

	//final decision
	final_decesion << <grid_dim3, block_dim3 >> >(Ind_J, Igv, s_dec, bias);
}

__global__ void scma_decode_main_kernel(int** b_dec, int** s_dec, cuComplex** y, cuComplex*** h, cuComplex*** CB, int N_s, double p_noise, 
	                                    float**** Igv, float**** Ivg, int** Ind_K, int** Ind_J ){
	Igv_Ivg_init(Igv, Ivg, N_s);

	int Nit = 4;
	dim3   grid_dim1(K, Dr, N_s);
	dim3   block_dim1(M*M, 1, 1);

	dim3   grid_dim2(N_s, J, 1);
	dim3   block_dim2(1, 1, 1);

	dim3   grid_dim3(N_s, J, 1);
	dim3   block_dim3(1, 1, 1);

	double ap = 1 / M;
	double EbN0 = 1 / (CB[0][0][1].abs()*CB[0][0][1].abs()* p_noise);
	double N0 = 1 / EbN0;

	int bias = 0;

	for (int iter = 0; iter < Nit; iter++){
		Igv_update6_l3_log_kernel << <grid_dim1, block_dim1 >> >(y, h, CB, Ind_K, Igv, Ivg, N0, bias);
		cudaDeviceSynchronize();

		Ivg_update_l2_log_kernel << <grid_dim2, block_dim2 >> >(Ind_J, Igv, Ivg, bias);
		cudaDeviceSynchronize();
	}

	//final decision
	final_decesion << <grid_dim3, block_dim3 >> >(Ind_J, Igv, s_dec, bias);

	cudaDeviceSynchronize();
	for (int j = 0; j < J; j++){
		dec2bit(b_dec[j], s_dec[j], N_s, 2);
	}


	//free_variables12(Ind_K, Ind_J, Igv, Ivg, s_dec, N_s);
}

//void scma_decode_main(int** b_dec, cuComplex** y, cuComplex*** h, cuComplex*** CB, int N_s, double p_noise){
//
//	int** s_dec = allocate_Symbols_dev(N_s);
//
//	cudaStream_t    stream0, stream1;
//
//	cudaStreamCreate(&stream0);
//	cudaStreamCreate(&stream1);
//
//	double ap = 1 / M;
//	double EbN0 = 1 / (CB[0][0][1].abs()*CB[0][0][1].abs()* p_noise);
//	double N0 = 1 / EbN0;
//
//	float**** Igv = allocatef_Igv(N_s);
//	float**** Ivg = allocatef_Ivg(N_s);
//
//	int** Ind_J = get_J_indices(CB, Dc);
//	int** Ind_K = get_K_indices(CB, Dr);
//
//	kernel_wrapup << <1, 1, 0, stream0 >> >(y, h, CB, N_s , Ind_K, Ind_J, Igv, Ivg, N0, s_dec, 0);
//
//	cudaDeviceSynchronize();
//	for (int j = 0; j < J; j++){
//		dec2bit(b_dec[j], s_dec[j], N_s, 2);
//	}
//
//	free_variables12(Ind_K, Ind_J, Igv, Ivg, s_dec, N_s);
//
//}