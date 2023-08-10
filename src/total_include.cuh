#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <random>

#include <math.h>
#include <stdlib.h>
#include <time.h>

#define imax(a,b) (a>b?a:b)

using namespace std;

#define K 4
#define M 4
#define J 6
#define Dr 3
#define Dc 2

__host__ __device__ struct cuComplex{
	double   r;
	double   i;
	__host__ __device__ cuComplex(double a, double b) : r(a), i(b)  {}
	__host__ __device__ double magnitude2(void) { return r * r + i * i; }
	__host__ __device__ double abs(void)  { return sqrt(r * r + i * i); }

	__host__ __device__  cuComplex operator*(const cuComplex& a) {
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}

	__host__ __device__  cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}

	__host__ __device__ cuComplex operator-(const cuComplex& a) {
		return cuComplex(r - a.r, i - a.i);
	}
};
