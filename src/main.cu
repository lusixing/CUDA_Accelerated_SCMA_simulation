#include"total_include.cuh"
#include"scma_cuda_test.cuh" 
#include"uniform.cuh"
#include"gaussian.cuh"
#include"space_allocate.cuh"
#include"scma_cuda_utils.cuh"
//#include"scma_decode_serial.cuh"

#include"scma_decode_cuda_v3.cuh"

void main(){
	cudaDeviceReset();
	//MemFree_test();
	//bias_test();

	cudaStream_t    stream0, stream1;

	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	size_t total_mem, free_mem;

	cuComplex*** CB = get_default_codebook();

	const int SNR_len = 6;
	double SNR_dB[SNR_len] = { 8, 10, 12, 14, 16, 18 };

	int Nerr1[SNR_len] = { 0, 0, 0, 0, 0, 0 };
	int Nerr2[SNR_len] = { 0, 0, 0, 0, 0, 0 };

	double BER_avg1[SNR_len] = { 0, 0, 0, 0, 0, 0 };
	double BER_avg2[SNR_len] = { 0, 0, 0, 0, 0, 0 };

	int N_b = 384;   //length of  bit sequence 

	int N_s = N_b / log2(M); //length of symbol sequence

	int f_num = 5;

	int** Bits1 = allocate_Bits_host(N_b);
	int** Symbols1 = allocate_Bits_host(N_s);
	int** b_dec1_dev = allocate_Bits_dev(N_b);
	int** s_dec1_dev = allocate_Symbols_dev(N_s);


	int** Bits2 = allocate_Bits_host(N_b);
	int** Symbols2 = allocate_Bits_host(N_s);
	int** b_dec2_dev = allocate_Bits_dev(N_b);
	int** s_dec2_dev = allocate_Symbols_dev(N_s);

	cuComplex*** h = allocate_h(N_s);             //size(h) =[J][N][K]

	cuComplex*** x1 = allocate_x_host(N_s);           //size(x) =[J][N][K]
	cuComplex** y1 = allocate_y(N_s);

	cuComplex*** x2 = allocate_x_host(N_s);         
	cuComplex** y2 = allocate_y(N_s);

	float**** Igv1 = allocatef_Igv(N_s);
	float**** Ivg1 = allocatef_Ivg(N_s);

	float**** Igv2 = allocatef_Igv(N_s);
	float**** Ivg2 = allocatef_Ivg(N_s);

	int** Ind_J = get_J_indices(CB, Dc);
	int** Ind_K = get_K_indices(CB, Dr);

	float   total_elapsedTime = 0;

	for (int s = 0; s < 6; s++){
		double p_noise = 1 / pow(10, SNR_dB[s] / 10);
		for (int f = 0; f < f_num; f++){
			//Igv_Ivg_init(Igv, Ivg, N_s);

			cudaEvent_t     start, stop;
			float           elapsedTime;

			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			//cudaEventRecord(start, 0);

			source_bit(Bits1, N_b);
			source_bit(Bits2, N_b);

			get_h(h, N_s);
			scma_encode(x1, CB, Bits1, Symbols1, N_s);
			scma_uplink_transmisson(y1, h, x1, N_s, p_noise);
			
			scma_encode(x2, CB, Bits2, Symbols2, N_s);
			scma_uplink_transmisson(y2, h, x2, N_s, p_noise);

			cudaEventRecord(start, 0);

			scma_decode_main_kernel << <1, 1, 0, stream0 >> >(b_dec1_dev, s_dec1_dev, y1, h, CB, N_s, p_noise, Igv1, Ivg1, Ind_K, Ind_J);
			scma_decode_main_kernel << <1, 1, 0, stream1 >> >(b_dec2_dev, s_dec2_dev, y2, h, CB, N_s, p_noise, Igv2, Ivg2, Ind_K, Ind_J);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsedTime, start, stop);
			printf("Time taken:  %3.1f ms\n", elapsedTime);
			total_elapsedTime += elapsedTime;

			cudaDeviceSynchronize();
			cudaStreamSynchronize(stream0);
			cudaStreamSynchronize(stream1);

			for (int j = 0; j < J; j++){
				for (int nb = 0; nb < N_b; nb++){
					if (b_dec1_dev[j][nb] != Bits1[j][nb]){
						Nerr1[s]++;
					}
					if (b_dec2_dev[j][nb] != Bits2[j][nb]){
						Nerr2[s]++;
					}
				}
			}

			//printf("%d\n",Nerr[s]);

			//free_variables22(x, h, y, N_s);

			//cudaMemGetInfo(&free_mem, &total_mem);
			//std::cout << " Currently " << free_mem << " bytes free" << std::endl;

		}
		BER_avg1[s] = (double)Nerr1[s] / (f_num * N_b * J);
		BER_avg2[s] = (double)Nerr2[s] / (f_num * N_b * J);
		cout << "average BER1 of SNR =" << SNR_dB[s] << "dB: " << BER_avg1[s] << endl;
		cout << "average BER2 of SNR =" << SNR_dB[s] << "dB: " << BER_avg2[s] << endl;
	}

	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);

	printf("total decoding throughtput:%f Kb/s", (2 * SNR_len*f_num*N_b*J) / total_elapsedTime);
	system("pause");

}