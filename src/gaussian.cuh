#pragma once

#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "total_include.cuh"

#define PI 3.141592653

double gaussian_pdf(double x, double mu, double sig){
	double g;
	g = (1 / sqrt(2 * PI*sig*sig))*exp(-(x - mu)*(x - mu) / (2 * sig*sig));
	return g;
}


void Gaussian_test(void)
{
	const int nrolls = 1000;  // number of experiments
	const int nstars = 200;    // maximum number of stars to distribute
	const int scale = 20;

	const double mu = 10;
	const double sig = 4;
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(mu, sig);

	int p[scale] = {};

	for (int i = 0; i<nrolls; ++i) {
		double number = distribution(generator);
		std::cout << number << std::endl;
		if ((number >= 0.0) && (number<(double)scale)) ++p[int(number)];
	}

	std::cout << "normal_distribution (mu,sigma):" << std::endl;

	for (int i = 0; i<scale; ++i) {
		std::cout << i << "-" << (i + 1) << ": ";
		std::cout << std::string(p[i] * nstars / nrolls, '*') << std::endl;
	}


}

double* Gaussian_noise(const int len, double mu, double sig, int type){

	double* noise;
	//if (type == 1){
	noise = (double*)malloc(len*sizeof(double));
	//}
	//else if (type==2){
	//	cudaMallocManaged(&noise, sizeof(double)*len);
	//}

	std::random_device rd;
	std::default_random_engine generator;
	generator.seed(rd());
	std::normal_distribution<double> distribution(mu, sig);

	for (int i = 0; i < len; i++){
		noise[i] = distribution(generator);
	}

	return noise;
}

double randn_test1(double mu, double sig)
{
	std::random_device rd;
	std::default_random_engine generator;
	generator.seed(rd());
	std::normal_distribution<double> distribution(mu, sig);

	double number = distribution(generator);

	return number;

}


double randn_test2(double mu, double sigma)
{

	double U1, U2, W, mult;
	static double X1, X2;
	static int call = 0;

	if (call == 1)
	{
		call = !call;
		return (mu + sigma * (double)X2);
	}

	do
	{
		U1 = -1 + ((double)rand() / RAND_MAX) * 2;
		U2 = -1 + ((double)rand() / RAND_MAX) * 2;
		W = pow(U1, 2) + pow(U2, 2);
	} while (W >= 1 || W == 0);

	mult = sqrt((-2 * log(W)) / W);
	X1 = U1 * mult;
	X2 = U2 * mult;

	call = !call;

	return (mu + sigma * (double)X1);
}

void AWGN2(cuComplex** x, int N_s, double p_noise){

	std::random_device rd;
	std::default_random_engine generator;
	generator.seed(rd());
	std::normal_distribution<double> distribution(0, p_noise);

	for (int n = 0; n < N_s; n++){
		for (int k = 0; k < K; k++){

			double noise_r = distribution(generator);
			double noise_i = distribution(generator);
			x[n][k].r += noise_r;
			x[n][k].i += noise_i;

			//	cout << noise_r << ' ' << noise_i << endl;
		}

	}
}

__host__ cuComplex complex_noise(double p_noise){
	cuComplex n(0, 0);

	std::random_device rd;
	std::default_random_engine generator;
	generator.seed(rd());
	std::normal_distribution<double> distribution(0, p_noise);

	double n_r = distribution(generator);
	double n_i = distribution(generator);

	n.r = n_r;
	n.i = n_i;

	return n;
}