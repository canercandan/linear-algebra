#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas.h>

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

void modify(float *m, int ldm, int n, int p, int q, float alpha, float beta)
{
    cublasSscal (n-p+1, alpha, &m[IDX2F(p,q,ldm)], ldm);
    cublasSscal (ldm-p+1, beta, &m[IDX2F(p,q,ldm)], 1);
}

#define M 6
#define N 5

int main(void)
{
    int i, j;
    cublasStatus stat;
    float* devPtrA;
    float* a = 0;

    a = (float *)malloc (M * N * sizeof (*a));
    if (!a)
	{
	    printf ("host memory allocation failed");
	    return EXIT_FAILURE;
	}

    for (j = 1; j <= N; j++)
	{
	    for (i = 1; i <= M; i++)
		{
		    a[IDX2F(i,j,M)] = (i-1) * M + j;
		}
	}

    cublasInit();

    stat = cublasAlloc (M*N, sizeof(*a), (void**)&devPtrA);
    if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("device memory allocation failed");
	    cublasShutdown();
	    return EXIT_FAILURE;
	}

    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("data download failed");
	    cublasFree (devPtrA);
	    cublasShutdown();
	    return EXIT_FAILURE;
	}

    modify (devPtrA, M, N, 2, 3, 16.0f, 12.0f);

    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS)
	{
	    printf ("data upload failed");
	    cublasFree (devPtrA);
	    cublasShutdown();
	    return EXIT_FAILURE;
	}

    cublasFree (devPtrA);
    cublasShutdown();

    for (j = 1; j <= N; j++)
	{
	    for (i = 1; i <= M; i++)
		{
		    printf ("%7.0f", a[IDX2F(i,j,M)]);
		}
	    printf ("\n");
	}

    free(a);

    return EXIT_SUCCESS;
}
