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
 * Aurèle Mahéo <aurele.maheo@gmail.com>
 */

#include <cstdio>

#include <cublas.h>

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int ComputePowerIteration(float* b, float* A, int N, float tol)
{
    printf("Compute power iteration..\n");

    //int k;

    float norm;

    float *d_A;
    float *d_b;

    int N2 = N*N;

    cublasStatus status;
    status = cublasAlloc(N2, sizeof(A), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS)
	{
	    fprintf (stderr, "!!!! CUBLAS initialization A error\n");
	    return -1;
	}

    status = cublasAlloc(N, sizeof(b), (void**)&d_b);
    if (status != CUBLAS_STATUS_SUCCESS)
	{
	    fprintf (stderr, "!!!! CUBLAS initialization b error\n");
	    return -1;
	}

    cublasSetMatrix(N, N, sizeof(A), A, N, d_A, N);
    cublasSetVector(N, sizeof(b), b, 1, d_b, 1);

    printf("N: %d\n", N);

    int i = 0;
    //int j;

    do
	{
	    i += 1;
	    printf("iteration %d\n", i);

	    //  for (j = 1; j <= N; j++) {
	    //   for (i = 1; i <= N; i++) {
	    //    printf("A %d: %f\n", IDX2F(i,j,N), A[IDX2F(i,j,N)]);
	    //   }
	    //  }

	    //   for(j=0;j<N;j++)
	    //  printf("b %d: %f\n", j, b[j]);

	    //Calculate norm
	    // float norm1 = cublasSnrm2(N, d_b, 1);
	    // printf("first norm: %f\n", norm1);

	    //Matrix Vector product
	    cublasSgemv('n', N, N, 1, d_A, N, d_b, 1, 0, d_b, 1);

	    //Calculate norm
	    norm = cublasSnrm2(N, d_b, 1);
	    printf("norm: %f\n", norm);

	    // y /= norm;
	    //Scalar
	    cublasSscal(N, 1/norm, d_b, 1);
	}
    while(norm < tol);

    printf("Power computed..\n");

    return 0;
}

int main(int argc, char* argv[])
{
    int N =  5;

    printf("*******Power iteration method********\n");

    float* A;
    float* b;
    //float *y;
    int i, j;

    A = (float*)malloc(N*N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    //y = (float*)malloc(N*sizeof(float));

    for (j = 1; j <= N; j++)
	{
	    for (i = 1; i <= N; i++)
		{
		    A[IDX2F(i,j,N)] = i * N + j + 1;
		}
	    b[j-1] = j+1;
	}

    // for (j = 1; j <= N; j++) {
    //  for (i = 1; i <= N; i++) {
    //   printf("A %d: %f\n", IDX2C(i,j,N), A[IDX2C(i,j,N)]);
    //  }
    // }

    for(j=0;j<N;j++)
	{
	    printf("b %d: %f\n", j, b[j]);
	}

    //Initialize CUBLAS
    printf("Init cuBLAS..\n");
    cublasInit();
    printf("cuBLAS initialized..\n");

    ComputePowerIteration(b, A, N, 1e5);

    return 0;
}
