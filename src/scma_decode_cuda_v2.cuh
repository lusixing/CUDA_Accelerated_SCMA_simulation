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

void get_f_l3_log(double*** f_k, cuComplex y_nk, cuComplex*** h, cuComplex*** CB, int* ind_k, int k, int n, double N0){        //size (CB) = [J][M][K]
	////size(h) =[J][N][K]          
	for (int m1 = 0; m1 < M; m1++){
		for (int m2 = 0; m2 < M; m2++){
			for (int m3 = 0; m3 < M; m3++){
				cuComplex diff = y_nk - (CB[ind_k[0]][m1][k] * h[ind_k[0]][n][k] + CB[ind_k[1]][m2][k] * h[ind_k[1]][n][k] + CB[ind_k[2]][m3][k] * h[ind_k[2]][n][k]);
				f_k[m1][m2][m3] = -(1 / N0) * diff.abs() * diff.abs();
			}

		}
	}

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

__global__ void get_f_l3_log_kernel(double**** f, cuComplex* y_n, cuComplex*** h, cuComplex*** CB, int** Ind_K, int n, double N0){        //size (CB) = [J][M][K]
	////size(h) =[J][N][K]          
	int k = blockIdx.x;
	int idx_linear = threadIdx.x;

	//printf("%running block %d thread %d\n", k, idx_linear);
	//__syncthreads();
	int* temp = linear_2_3D(idx_linear);
	//printf("returned index : %d%d%d",temp[0],temp[1],temp[2]);
	//__syncthreads();

	int m1 = temp[0];
	int m2 = temp[1];
	int m3 = temp[2];

	cuComplex diff = y_n[k] - (CB[Ind_K[k][0]][m1][k] * h[Ind_K[k][0]][n][k] + CB[Ind_K[k][1]][m2][k] * h[Ind_K[k][1]][n][k] + CB[Ind_K[k][2]][m3][k] * h[Ind_K[k][2]][n][k]);

	f[k][m1][m2][m3] = -(1 / N0) * diff.abs() * diff.abs();

	//**printf("%f", f[k][m1][m2][m3]);

	//printf("finished\n");
	//__syncthreads();
}

void Igv_update_l3_log(double*** f_k, int* ind_k, double*** Igv, double*** Ivg, int k){

	for (int m1 = 0; m1 < M; m1++){
		double sIgv[M*M] = { 0 };
		int idx = 0;
		for (int m2 = 0; m2 < M; m2++){
			for (int m3 = 0; m3 < M; m3++){
				sIgv[idx] = f_k[m1][m2][m3] + Ivg[k][ind_k[1]][m2] + Ivg[k][ind_k[2]][m3];

				idx++;
			}
		}
		double lsp = log_sum_exp(sIgv, M*M);
		Igv[k][ind_k[0]][m1] = lsp;

	}

	for (int m2 = 0; m2 < M; m2++){
		double sIgv[M*M] = { 0 };
		int idx = 0;
		for (int m1 = 0; m1 < M; m1++){
			for (int m3 = 0; m3 < M; m3++){
				sIgv[idx] = f_k[m1][m2][m3] + Ivg[k][ind_k[0]][m1] + Ivg[k][ind_k[2]][m3];
				idx++;
			}
		}

		double lsp = log_sum_exp(sIgv, M*M);
		Igv[k][ind_k[1]][m2] = lsp;

	}

	for (int m3 = 0; m3 < M; m3++){
		double sIgv[M*M] = { 0 };
		int idx = 0;
		for (int m1 = 0; m1 < M; m1++){
			for (int m2 = 0; m2 < M; m2++){
				sIgv[idx] = f_k[m1][m2][m3] + Ivg[k][ind_k[0]][m1] + Ivg[k][ind_k[1]][m2];
				idx++;
			}
		}

		double lsp = log_sum_exp(sIgv, M*M);
		Igv[k][ind_k[2]][m3] = lsp;

	}
}

void Igv_update2_l3_log(cuComplex* y_n, cuComplex*** h, cuComplex*** CB, int** Ind_K, double*** Igv, double*** Ivg, int n, double N0){

	for (int k = 0; k < K; k++){
		for (int m1 = 0; m1 < M; m1++){
			double sIgv[M*M] = { 0 };
			int idx = 0;
			for (int m2 = 0; m2 < M; m2++){
				for (int m3 = 0; m3 < M; m3++){
					cuComplex diff = y_n[k] - (CB[Ind_K[k][0]][m1][k] * h[Ind_K[k][0]][n][k] + CB[Ind_K[k][1]][m2][k] * h[Ind_K[k][1]][n][k] + CB[Ind_K[k][2]][m3][k] * h[Ind_K[k][2]][n][k]);
					double f_temp = -(1 / N0) * diff.abs() * diff.abs();

					sIgv[idx] = f_temp + Ivg[k][Ind_K[k][1]][m2] + Ivg[k][Ind_K[k][2]][m3];
					idx++;
				}
			}
			double lsp = log_sum_exp(sIgv, M*M);
			Igv[k][Ind_K[k][0]][m1] = lsp;

		}

		for (int m2 = 0; m2 < M; m2++){
			double sIgv[M*M] = { 0 };
			int idx = 0;
			for (int m1 = 0; m1 < M; m1++){
				for (int m3 = 0; m3 < M; m3++){
					cuComplex diff = y_n[k] - (CB[Ind_K[k][0]][m1][k] * h[Ind_K[k][0]][n][k] + CB[Ind_K[k][1]][m2][k] * h[Ind_K[k][1]][n][k] + CB[Ind_K[k][2]][m3][k] * h[Ind_K[k][2]][n][k]);
					double f_temp = -(1 / N0) * diff.abs() * diff.abs();

					sIgv[idx] = f_temp + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][2]][m3];
					idx++;
				}
			}

			double lsp = log_sum_exp(sIgv, M*M);
			Igv[k][Ind_K[k][1]][m2] = lsp;

		}

		for (int m3 = 0; m3 < M; m3++){
			double sIgv[M*M] = { 0 };
			int idx = 0;
			for (int m1 = 0; m1 < M; m1++){
				for (int m2 = 0; m2 < M; m2++){
					cuComplex diff = y_n[k] - (CB[Ind_K[k][0]][m1][k] * h[Ind_K[k][0]][n][k] + CB[Ind_K[k][1]][m2][k] * h[Ind_K[k][1]][n][k] + CB[Ind_K[k][2]][m3][k] * h[Ind_K[k][2]][n][k]);
					double f_temp = -(1 / N0) * diff.abs() * diff.abs();

					sIgv[idx] = f_temp + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][1]][m2];
					idx++;
				}
			}

			double lsp = log_sum_exp(sIgv, M*M);
			Igv[k][Ind_K[k][2]][m3] = lsp;
		}
	}

}


__global__ void Igv_update2_l3_log_kernel(double**** f, int** Ind_K, double*** Igv, double*** Ivg){
	int k = blockIdx.x;
	int idx_dr = blockIdx.y;
	int idx_m = blockIdx.z;

	double temp;

	switch (idx_dr){

	case 0:
		int m1 = idx_m;
		temp = 0;
		for (int m2 = 0; m2 < M; m2++){
			for (int m3 = 0; m3 < M; m3++){
				temp += exp(f[k][m1][m2][m3] + Ivg[k][Ind_K[k][1]][m2] + Ivg[k][Ind_K[k][2]][m3]);
			}
		}
		Igv[k][Ind_K[k][0]][m1] = log(temp);

	case 1:
		int m2 = idx_m;
		temp = 0;
		for (int m1 = 0; m1 < M; m1++){
			for (int m3 = 0; m3 < M; m3++){
				temp += exp(f[k][m1][m2][m3] + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][2]][m3]);
			}
		}
		Igv[k][Ind_K[k][1]][m2] = log(temp);

	case 2:
		int m3 = idx_m;
		temp = 0;
		for (int m1 = 0; m1 < M; m1++){
			for (int m2 = 0; m2 < M; m2++){
				temp += exp(f[k][m1][m2][m3] + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][1]][m2]);
			}
		}
		Igv[k][Ind_K[k][2]][m3] = log(temp);
	}
}

__global__ void Igv_update3_l3_log_kernel(double**** f, int** Ind_K, double*** Igv, double*** Ivg){
	int k = blockIdx.x;
	int idx_dr = blockIdx.y;

	double temp;

	int m1;
	int m2;
	int m3;
	int idx_temp;
	__shared__ double Sigv[M*M];

	switch (idx_dr){

	case 0:
		m2 = threadIdx.x;
		m3 = threadIdx.y;
		idx_temp = m2*M + m3;
		for (int m1 = 0; m1 < M; m1++){
			
			Sigv[idx_temp] = exp(f[k][m1][m2][m3] + Ivg[k][Ind_K[k][1]][m2] + Ivg[k][Ind_K[k][2]][m3]);
				
			__syncthreads();
			int i = M*M / 2;
			while (i != 0) {
				if (idx_temp < i)
					Sigv[idx_temp] += Sigv[idx_temp + i];
				__syncthreads();
				i /= 2;
			}

			if (idx_temp == 0)
				Igv[k][Ind_K[k][0]][m1] = log(Sigv[0]);

			//Igv[k][Ind_K[k][0]][m1] = log(temp);
		}


	case 1:
		for (int m2 = 0; m2 < M; m2++){
			temp = 0;
			m1 = threadIdx.x;
			m3 = threadIdx.y;
			idx_temp = m1*M + m3;

			Sigv[idx_temp] = exp(f[k][m1][m2][m3] + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][2]][m3]);
				
			__syncthreads();

			int i = M*M / 2;
			while (i != 0) {
				if (idx_temp < i)
					Sigv[idx_temp] += Sigv[idx_temp + i];
				__syncthreads();
				i /= 2;
			}

			if (idx_temp == 0)
				Igv[k][Ind_K[k][1]][m2] =log( Sigv[0]);


			//Igv[k][Ind_K[k][1]][m2] = log(temp);
		}


	case 2:
		for (int m3 = 0; m3 < M; m3++){
			temp = 0;
			m1 = threadIdx.x;
			m2 = threadIdx.y;
			idx_temp = m1*M + m2;

			Sigv[idx_temp] = exp(f[k][m1][m2][m3] + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][1]][m2]);
				
			__syncthreads();

			int i = M*M / 2;
			while (i != 0) {
				if (idx_temp < i)
					Sigv[idx_temp] += Sigv[idx_temp + i];
				__syncthreads();
				i /= 2;
			}

			if (idx_temp == 0)
				Igv[k][Ind_K[k][2]][m3] =log( Sigv[0]);

			//Igv[k][Ind_K[k][2]][m3] = log(temp);
		}

	}
}

__global__ void Igv_update4_l3_log_kernel(double**** f, int** Ind_K, double*** Igv, double*** Ivg){
	int k = blockIdx.x;
	int idx_dr = blockIdx.y;

	double temp;

	int m1;
	int m2;
	int m3;
	int idx_temp;
	int* idx_2d;
	__shared__ double Sigv[M*M];

	switch (idx_dr){

	case 0:
		idx_temp = threadIdx.x;
		idx_2d = linear_2_2D(idx_temp);
		m2 = idx_2d[0];
		m3 = idx_2d[1];
		//idx_temp = m2*M + m3;
		for (int m1 = 0; m1 < M; m1++){

			Sigv[idx_temp] = exp(f[k][m1][m2][m3] + Ivg[k][Ind_K[k][1]][m2] + Ivg[k][Ind_K[k][2]][m3]);

			__syncthreads();
			int i = M*M / 2;
			while (i != 0) {
				if (idx_temp < i)
					Sigv[idx_temp] += Sigv[idx_temp + i];
				__syncthreads();
				i /= 2;
			}

			if (idx_temp == 0)
				Igv[k][Ind_K[k][0]][m1] = log(Sigv[0]);
		}


	case 1:
		idx_temp = threadIdx.x;
		idx_2d = linear_2_2D(idx_temp);
		m1 = idx_2d[0];
		m3 = idx_2d[1];

		for (int m2 = 0; m2 < M; m2++){
			Sigv[idx_temp] = exp(f[k][m1][m2][m3] + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][2]][m3]);

			__syncthreads();

			int i = M*M / 2;
			while (i != 0) {
				if (idx_temp < i)
					Sigv[idx_temp] += Sigv[idx_temp + i];
				__syncthreads();
				i /= 2;
			}

			if (idx_temp == 0)
				Igv[k][Ind_K[k][1]][m2] = log(Sigv[0]);
		}


	case 2:
		idx_temp = threadIdx.x;
		idx_2d = linear_2_2D(idx_temp);
		m1 = idx_2d[0];
		m2 = idx_2d[1];

		for (int m3 = 0; m3 < M; m3++){

			Sigv[idx_temp] = exp(f[k][m1][m2][m3] + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][1]][m2]);

			__syncthreads();

			int i = M*M / 2;
			while (i != 0) {
				if (idx_temp < i)
					Sigv[idx_temp] += Sigv[idx_temp + i];
				__syncthreads();
				i /= 2;
			}

			if (idx_temp == 0)
				Igv[k][Ind_K[k][2]][m3] = log(Sigv[0]);
		}

	}
}

__global__ void Igv_update5_l3_log_kernel(cuComplex* y_n, cuComplex*** h, cuComplex*** CB, int** Ind_K, double*** Igv, double*** Ivg, double N0,int n){
	int k = blockIdx.x;
	int idx_dr = blockIdx.y;

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
			diff = y_n[k] - (CB[Ind_K[k][0]][m1][k] * h[Ind_K[k][0]][n][k] + CB[Ind_K[k][1]][m2][k] * h[Ind_K[k][1]][n][k] + CB[Ind_K[k][2]][m3][k] * h[Ind_K[k][2]][n][k]);
			f_temp = -(1 / N0) * diff.abs() * diff.abs();

			 Sigv[idx_temp] = exp(f_temp + Ivg[k][Ind_K[k][1]][m2] + Ivg[k][Ind_K[k][2]][m3]);
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
				Igv[k][Ind_K[k][0]][m1] = log(Sigv[0]);
		}


	case 1:
		idx_temp = threadIdx.x;
		idx_2d = linear_2_2D(idx_temp);
		m1 = idx_2d[0];
		m3 = idx_2d[1];

		for (int m2 = 0; m2 < M; m2++){
			diff = y_n[k] - (CB[Ind_K[k][0]][m1][k] * h[Ind_K[k][0]][n][k] + CB[Ind_K[k][1]][m2][k] * h[Ind_K[k][1]][n][k] + CB[Ind_K[k][2]][m3][k] * h[Ind_K[k][2]][n][k]);
			f_temp = -(1 / N0) * diff.abs() * diff.abs();

			Sigv[idx_temp] = exp(f_temp + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][2]][m3]);
			//Sigv[idx_temp] = exp(f[k][m1][m2][m3] + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][2]][m3]);

			__syncthreads();

			int i = M*M / 2;
			while (i != 0) {
				if (idx_temp < i)
					Sigv[idx_temp] += Sigv[idx_temp + i];
				__syncthreads();
				i /= 2;
			}

			if (idx_temp == 0)
				Igv[k][Ind_K[k][1]][m2] = log(Sigv[0]);
		}


	case 2:
		idx_temp = threadIdx.x;
		idx_2d = linear_2_2D(idx_temp);
		m1 = idx_2d[0];
		m2 = idx_2d[1];

		for (int m3 = 0; m3 < M; m3++){
			diff = y_n[k] - (CB[Ind_K[k][0]][m1][k] * h[Ind_K[k][0]][n][k] + CB[Ind_K[k][1]][m2][k] * h[Ind_K[k][1]][n][k] + CB[Ind_K[k][2]][m3][k] * h[Ind_K[k][2]][n][k]);
			f_temp = -(1 / N0) * diff.abs() * diff.abs();

			Sigv[idx_temp] = exp(f_temp + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][1]][m2]);
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
				Igv[k][Ind_K[k][2]][m3] = log(Sigv[0]);
		}

	}
}

__global__ void Igv_update6_l3_log_kernel(cuComplex** y, cuComplex*** h, cuComplex*** CB, int** Ind_K, double*** Igv, double*** Ivg, double N0){
	int k = blockIdx.x;
	int idx_dr = blockIdx.y;
	int n = blockIdx.z;

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

			Sigv[idx_temp] = exp(f_temp + Ivg[k][Ind_K[k][1]][m2] + Ivg[k][Ind_K[k][2]][m3]);
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
				Igv[k][Ind_K[k][0]][m1] = log(Sigv[0]);
		}


	case 1:
		idx_temp = threadIdx.x;
		idx_2d = linear_2_2D(idx_temp);
		m1 = idx_2d[0];
		m3 = idx_2d[1];

		for (int m2 = 0; m2 < M; m2++){
			diff = y[n][k] - (CB[Ind_K[k][0]][m1][k] * h[Ind_K[k][0]][n][k] + CB[Ind_K[k][1]][m2][k] * h[Ind_K[k][1]][n][k] + CB[Ind_K[k][2]][m3][k] * h[Ind_K[k][2]][n][k]);
			f_temp = -(1 / N0) * diff.abs() * diff.abs();

			Sigv[idx_temp] = exp(f_temp + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][2]][m3]);
			//Sigv[idx_temp] = exp(f[k][m1][m2][m3] + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][2]][m3]);

			__syncthreads();

			int i = M*M / 2;
			while (i != 0) {
				if (idx_temp < i)
					Sigv[idx_temp] += Sigv[idx_temp + i];
				__syncthreads();
				i /= 2;
			}

			if (idx_temp == 0)
				Igv[k][Ind_K[k][1]][m2] = log(Sigv[0]);
		}


	case 2:
		idx_temp = threadIdx.x;
		idx_2d = linear_2_2D(idx_temp);
		m1 = idx_2d[0];
		m2 = idx_2d[1];

		for (int m3 = 0; m3 < M; m3++){
			diff = y[n][k] - (CB[Ind_K[k][0]][m1][k] * h[Ind_K[k][0]][n][k] + CB[Ind_K[k][1]][m2][k] * h[Ind_K[k][1]][n][k] + CB[Ind_K[k][2]][m3][k] * h[Ind_K[k][2]][n][k]);
			f_temp = -(1 / N0) * diff.abs() * diff.abs();

			Sigv[idx_temp] = exp(f_temp + Ivg[k][Ind_K[k][0]][m1] + Ivg[k][Ind_K[k][1]][m2]);
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
				Igv[k][Ind_K[k][2]][m3] = log(Sigv[0]);
		}

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

	int** s_dec = allocate_Symbols(N_s);

	int Nit = 5;

	double ap = 1 / M;
	double EbN0 = 1 / (CB[0][0][1].abs()*CB[0][0][1].abs()* p_noise);
	double N0 = 1 / EbN0;

	//double*** Igv_n = allocate_Igv_n();
	//double*** Ivg_n = allocate_Ivg_n(ap);
	double**** Igv = allocate_Igv(N_s);
	double**** Ivg = allocate_Ivg(N_s,ap);

	//double**** f = allocate_f();

	int** Ind_J = get_J_indices(CB, Dc);
	int** Ind_K = get_K_indices(CB, Dr );

	for (int n = 0; n < N_s; n++){
		double*** Igv_n = Igv[n];
		double*** Ivg_n = Ivg[n];

		Igv_Ivg_init(Igv_n, Ivg_n);

	//	for (int k = 0; k < K; k++){
	//		get_f_l3_log(f[k], y[n][k], h, CB, Ind_K[k], k, n, N0);         //size(y) = [N_s][K]
	//	}

		//get_f_l3_log_kernel << <K, pow(M, 3) >> >(f, y[n], h, CB, Ind_K, n, N0);

		//cudaDeviceSynchronize();
		dim3   grid_dim(K, Dr, 1);
		dim3   block_dim(M*M, 1, 1);

		for (int iter = 0; iter < Nit; iter++){
			//for (int k = 0; k < K; k++){
			//	Igv_update_l3_log(f[k], Ind_K[k], Igv, Ivg, k);
			//}
			

			//Igv_update_l3_log_kernel << <grid_dim, M*M >> >(f, Ind_K, Igv, Ivg);
			//Igv_update4_l3_log_kernel << <grid_dim, block_dim >> >(f, Ind_K, Igv_n, Ivg_n);
			Igv_update5_l3_log_kernel << <grid_dim, block_dim >> >( y[n], h,  CB,  Ind_K,  Igv_n,  Ivg_n,  N0, n);

			cudaDeviceSynchronize();

			for (int j = 0; j < J; j++){
				Ivg_update_l2_log(Ind_J[j], Igv_n, Ivg_n, j);
			}
		}

		//final decision
		for (int j = 0; j < J; j++){
			double Q_max = -99999;
			//int max_ind = 0;
			for (int m = 0; m < M; m++){

				double Q_m = 0;
				for (int i = 0; i < Dc; i++){
					Q_m += Igv_n[Ind_J[j][i]][j][m];
				}
				if (Q_m > Q_max){
					Q_max = Q_m;
					s_dec[j][n] = m;
				}
			}
		}

	}


	for (int j = 0; j < J; j++){
		dec2bit(b_dec[j], s_dec[j], N_s, 2);
	}

	free_variables12(Ind_K, Ind_J, Igv, Ivg, s_dec, N_s);

}