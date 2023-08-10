#pragma once
#include "total_include.cuh"

int uniform_test1()
{
	const int nrolls = 100000;  // number of experiments
	const int nstars = 95;     // maximum number of stars to distribute
	const int nintervals = 10; // number of intervals

	std::random_device rd;
	std::default_random_engine generator;
	generator.seed(rd());
	std::uniform_real_distribution<double> distribution(0.0, 1.0);

	int p[nintervals] = {};

	for (int i = 0; i<nrolls; ++i) {
		double number = distribution(generator);
		++p[int(nintervals*number)];
		//std::cout << number << endl;
	}

	std::cout << "uniform_real_distribution (0.0,1.0):" << std::endl;
	std::cout << std::fixed; std::cout.precision(1);

	for (int i = 0; i<nintervals; ++i) {
		std::cout << float(i) / nintervals << "-" << float(i + 1) / nintervals << ": ";
		std::cout << std::string(p[i] * nstars / nrolls, '*') << std::endl;
	}

	return 0;
}

void randint(int* nums, int min, int max, int len)
{
	std::random_device rd;
	std::default_random_engine generator;
	generator.seed(rd());
	std::uniform_real_distribution<double> distribution((double)min - 0.5, (double)max + 0.5);


	for (int i = 0; i<len; ++i) {
		double number = distribution(generator);
		if (number < min){ nums[i] = min; }
		if (number > max){ nums[i] = max; }

		for (int j = min; j <= max; j++){
			if (abs(number - j) <= 0.5)
				nums[i] = j;
		}
	}

}

void source_bit(int** bits, int N_b){
	for (int j = 0; j < J; j++){
		randint(bits[j], 0, 1, N_b);
	}

}