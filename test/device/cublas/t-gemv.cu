// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/* This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors:
 * Caner Candan <caner@candan.fr>, http://caner.candan.fr
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas.h>

#include <linear_algebra/linear_algebra>
#include <linear_algebra/detail/device/cublas/cublas>

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

using namespace linear_algebra::detail::device::cublas;

typedef float T;
//typedef double T;

int main(void)
{
    const int N = 10;
    const int M = 10;

    T* a = (float *)malloc (M * N * sizeof (*a));
    if (!a)
	{
	    printf ("host memory allocation failed");
	    return EXIT_FAILURE;
	}

    T* x = (float *)malloc (N * sizeof (*x));
    if (!x)
	{
	    printf ("host memory allocation failed");
	    return EXIT_FAILURE;
	}

    for (int j = 1; j <= N; j++)
	{
	    for (int i = 1; i <= M; i++)
		{
		    a[IDX2F(i,j,M)] = 2;
		}
	}

    for (int j = 0; j < N; j++)
	{
	    x[j] = 1;
	}

    cublasInit();

    cublasStatus stat;

    float* devPtrA;
    stat = cublasAlloc (M*N, sizeof(*a), (void**)&devPtrA);
    if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("device memory allocation failed");
	    cublasShutdown();
	    return EXIT_FAILURE;
	}

    stat = cublasSetVector (M*N, sizeof(*a), a, 1, devPtrA, 1);
    if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("data download failed");
	    cublasFree (devPtrA);
	    cublasShutdown();
	    return EXIT_FAILURE;
	}

    float* devPtrx;
    stat = cublasAlloc (N, sizeof(*x), (void**)&devPtrx);
    if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("device memory allocation failed");
	    cublasShutdown();
	    return EXIT_FAILURE;
	}

    stat = cublasSetVector (N, sizeof(*x), a, 1, devPtrx, 1);
    if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("data download failed");
	    cublasFree (devPtrx);
	    cublasShutdown();
	    return EXIT_FAILURE;
	}

    float* devPtry;
    stat = cublasAlloc (N, sizeof(*devPtry), (void**)&devPtry);
    if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("device memory allocation failed");
	    cublasShutdown();
	    return EXIT_FAILURE;
	}

    cublasSgemv('n', M, N, 1, devPtrA, M, devPtrx, 1, 0, devPtry, 1);
    stat = cublasGetError();
    if ( stat != CUBLAS_STATUS_SUCCESS )
	{
	    throw std::runtime_error("gemv failed");
	}

    T* y = (float *)malloc (M * sizeof (*y));
    if (!y)
	{
	    printf ("host memory allocation failed");
	    return EXIT_FAILURE;
	}

    stat = cublasGetVector (M, sizeof(*y), devPtry, 1, y, 1);
    if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("data download failed");
	    cublasFree (devPtry);
	    cublasShutdown();
	    return EXIT_FAILURE;
	}

    printf("[");
    for (int i = 0; i < M; i++)
	{
	    printf("%f ", y[i]);
	}
    printf("]\n");

    cublasShutdown();

    return 0;
}
