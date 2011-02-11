#include <cstring>

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil.h>

#include "GPGPU.hpp"

#include "cudacomplex.h"

void InitGPU(int device)
{
    InitCUDA(device);
}

void InitCUDA(int device)
{
    ///////////////////////////
    // CUDA initialisation
    ///////////////////////////

    int deviceCount;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) std::cout << "There is no device supporting CUDA" << std::endl;

    CUDA_SAFE_CALL(cudaSetDevice(device));
    cudaDeviceProp deviceProp;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));
    std::cout << "Device " << device << ": " << deviceProp.name << std::endl;

    // or
    // CUT_DEVICE_INIT(); // with --device=1 (num device chosen)
}

template <class T>
void mat_vec_csr(Matrix_CSR<T> *A, T *x, T *b)
{
    T temp;
    for(int i=0;i<A->n;++i){
	temp=0;
	for(int k=A->row_index[i];k<A->row_index[i+1];++k)
	    {
		T t = x[A->col[k]] * A->val[k];
		temp = temp + t;
	    }
	b[i] = temp;
    }
}

template void mat_vec_csr(Matrix_CSR<double> *A, double *x, double *b);
template void mat_vec_csr(Matrix_CSR<float> *A, float *x, float *b);

template void mat_vec_csr(Matrix_CSR< doublecomplex > *A, doublecomplex *x, doublecomplex *b);
template void mat_vec_csr(Matrix_CSR< singlecomplex > *A, singlecomplex *x, singlecomplex *b);

template <class T>
void add_mul(T *u, T *v, T alpha, T *w, int size)
{
    if(u!=NULL && v!=NULL){
	for(int i=0;i<size;++i)
	    w[i] = u[i] + alpha * v[i];
    }else{
	if(v==NULL){
	    for(int i=0;i<size;++i)
		w[i] = u[i];
	}else{ //-- u==NULL
	    for(int i=0;i<size;++i)
		w[i] = alpha * v[i];
	}
    }
}

template void add_mul(double *u, double *v, double alpha, double *w, int size);
template void add_mul(float *u, float *v, float alpha, float *w, int size);

template void add_mul(doublecomplex *u, doublecomplex *v, doublecomplex alpha, doublecomplex *w, int size);
template void add_mul(singlecomplex *u, singlecomplex *v, singlecomplex alpha, singlecomplex *w, int size);

template <class T>
void norm(T *u, T* norm,int size)
{
    T temp=0;
    for(int i=0;i<size;++i)
	{
	    T t = u[i]*u[i];
	    temp = temp + t;
	}
    *norm=temp;
}

template void norm(double *u, double *norm, int size);
template void norm(float *u, float *norm, int size);

template void norm(doublecomplex *u, doublecomplex *norm, int size);
template void norm(singlecomplex *u, singlecomplex *norm, int size);

template <class T>
void dot(T *u, T *v, T* dot ,int size)
{
    T temp=0;
    for(int i=0;i<size;++i)
	{
	    T t = u[i]*v[i];
	    temp = temp + t;
	}
    *dot=temp;
}

template void dot(double *u, double *v ,double *dot, int size);
template void dot(float *u, float *v, float *dot, int size);

template void dot(doublecomplex *u, doublecomplex *v ,doublecomplex *dot, int size);
template void dot(singlecomplex *u, singlecomplex *v, singlecomplex *dot, int size);
