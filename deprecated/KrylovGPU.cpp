#include <complex>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil.h>

#include "KrylovGPU.hpp"

#include "cudacomplex.h"

template <class T>
KrylovGPU<T>::KrylovGPU(int n, T* val_rhs, int nnz, int* row_index, int* col, T* val_mat)
{
    ///////////////////////////////
    // Data initialization on GPU
    ///////////////////////////////

    m_n=n;
    m_nnz=nnz;
    m_mat.n=n;
    m_mat.nnz=nnz;
    m_rhs.n=n;
    m_x.n=n;
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_mat.row_index, (n+1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_mat.col, nnz * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_mat.val, nnz * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_x.val, n * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&m_rhs.val, n * sizeof(T)));

    /*
      T testb=T(0);
      for(int i=0;i<n;++i)
      testb+=val_rhs[i]*val_rhs[i];
      printf("--------------%d %e\n",n,testb);
    */

    ///////////////////////////////
    // Data copy to GPU
    ///////////////////////////////

    CUDA_SAFE_CALL(cudaMemcpy(m_mat.row_index, row_index, (n+1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(m_mat.col, col, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(m_mat.val, val_mat, nnz * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(m_rhs.val, val_rhs, n * sizeof(T), cudaMemcpyHostToDevice));

    LinAlgCUDA::init(m_x.val,T(0),m_n);
}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
KrylovGPU<T>::~KrylovGPU()
{
    ///////////////////////////////
    // Cleanup
    ///////////////////////////////

    CUDA_SAFE_CALL(cudaFree(m_mat.row_index));
    CUDA_SAFE_CALL(cudaFree(m_mat.col));
    CUDA_SAFE_CALL(cudaFree(m_mat.val));
    CUDA_SAFE_CALL(cudaFree(m_x.val));
    CUDA_SAFE_CALL(cudaFree(m_rhs.val));
#ifdef DEBUG
    CUT_CHECK_ERROR("Pb after GPU call");
#endif
}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovGPU<T>::SolveCGv1_CSR(T* res, int* nbIter, T tol)
{
    int nbIterMax=(*nbIter);

    ///////////////////////////////
    // Initialization
    ///////////////////////////////

    CUDA_SAFE_CALL(cudaMemcpy(m_x.val,res, m_n * sizeof(T), cudaMemcpyHostToDevice));

    ///////////////////////////////
    // Conjugate Gradient
    ///////////////////////////////

    // Initialization
    T rho, rho_num, rho_denum, test, stop_test,rhs;
    stop_test=tol*tol;
    T *r, *w, *temp, *norm, *dot;
    CUDA_SAFE_CALL(cudaMalloc((void**)&r, m_n * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&w, m_n * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&temp, m_n * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&norm, 1 * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dot, 2 * sizeof(T)));

    LinAlgCUDA::norm(m_rhs.val,norm,m_n);
    CUDA_SAFE_CALL(cudaMemcpy(&rhs, norm, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    if(rhs==0) rhs=1;
    if(rhs<stop_test) rhs=stop_test;

    // First step
    LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,m_x.val,temp,m_n); //-- temp = A*x
    LinAlgCUDA::add_mul(m_rhs.val, temp, T(-1), r, m_n); //-- r = b - A*x

    *nbIter=0;
    LinAlgCUDA::norm(r,norm,m_n);
    CUDA_SAFE_CALL(cudaMemcpy(&test, norm, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    if (test/rhs < stop_test) return;

    CUDA_SAFE_CALL(cudaMemcpy(w,r,m_n*sizeof(T),cudaMemcpyDeviceToDevice)); //-- w = r

    LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,w,temp,m_n); //-- temp = A*w

    LinAlgCUDA::dot_by2(r,w,&dot[0],w,temp,&dot[1],m_n);
    CUDA_SAFE_CALL(cudaMemcpy(&rho_num, &dot[0], 1 * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&rho_denum, &dot[1], 1 * sizeof(T), cudaMemcpyDeviceToHost));
    rho=rho_num/rho_denum;//-- rho = (r'*w)/(w'*(A*w))

    LinAlgCUDA::add_mul(m_x.val, w, rho, m_x.val, m_n); //-- x = x + rho*w

    // Loop
    *nbIter=1;
    while((*nbIter)<nbIterMax)
	{
	    LinAlgCUDA::add_mul(r, temp, -rho, r, m_n); //-- r = r - rho*A*w

	    // Stop test
	    LinAlgCUDA::norm(r,norm,m_n);
	    CUDA_SAFE_CALL(cudaMemcpy(&test, norm, 1 * sizeof(T), cudaMemcpyDeviceToHost));
	    if (test/rhs < stop_test) break;

	    LinAlgCUDA::dot_by2(r,temp,&dot[0],w,temp,&dot[1],m_n);
	    CUDA_SAFE_CALL(cudaMemcpy(&rho_num, &dot[0], 1 * sizeof(T), cudaMemcpyDeviceToHost));
	    CUDA_SAFE_CALL(cudaMemcpy(&rho_denum, &dot[1], 1 * sizeof(T), cudaMemcpyDeviceToHost));
	    rho=rho_num/rho_denum;//-- rho = (r'*A*w)/(w'*(A*w))
	    LinAlgCUDA::add_mul(r, w, -rho, w, m_n); //-- w = r - rho*w

	    LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,w,temp,m_n); //-- temp = A*w

	    LinAlgCUDA::dot_by2(r,w,&dot[0],w,temp,&dot[1],m_n);
	    CUDA_SAFE_CALL(cudaMemcpy(&rho_num, &dot[0], 1 * sizeof(T), cudaMemcpyDeviceToHost));
	    CUDA_SAFE_CALL(cudaMemcpy(&rho_denum, &dot[1], 1 * sizeof(T), cudaMemcpyDeviceToHost));
	    rho=rho_num/rho_denum;//-- rho = (r'*w)/(w'*(A*w))
	    LinAlgCUDA::add_mul(m_x.val, w, rho, m_x.val, m_n); //-- x = x + rho*w

	    (*nbIter)++;
	}

    // Cleanup
    CUDA_SAFE_CALL(cudaFree(w));
    CUDA_SAFE_CALL(cudaFree(r));
    CUDA_SAFE_CALL(cudaFree(temp));
    CUDA_SAFE_CALL(cudaFree(norm));
    CUDA_SAFE_CALL(cudaFree(dot));

    ///////////////////////////////
    // Copy solution to CPU
    ///////////////////////////////

    CUDA_SAFE_CALL(cudaMemcpy(res, m_x.val, m_n * sizeof(T), cudaMemcpyDeviceToHost));
}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovGPU<T>::SolveCGv2_CSR(T* res, int* nbIter, T tol)
{
    int nbIterMax=(*nbIter);

    ///////////////////////////////
    // Initialization
    ///////////////////////////////

    //LinAlgCUDA::init(m_x.val,T(0),m_n);
    CUDA_SAFE_CALL(cudaMemcpy(m_x.val,res, m_n * sizeof(T), cudaMemcpyHostToDevice));

    ///////////////////////////////
    // Conjugate Gradient
    ///////////////////////////////

    // Initialization
    T rho,rho_new, rho_old, rho_denum, stop_test, rhs;
    stop_test=tol*tol;
    T *r, *w, *temp, *norm, *dot;
    CUDA_SAFE_CALL(cudaMalloc((void**)&r, m_n * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&w, m_n * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&temp, m_n * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&norm, 1 * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dot, 1 * sizeof(T)));

    LinAlgCUDA::norm(m_rhs.val,dot,m_n);
    CUDA_SAFE_CALL(cudaMemcpy(&rhs, dot, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    if(rhs==0) rhs=1;
    if(rhs<stop_test) rhs=stop_test;

    // First step
    LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,m_x.val,temp,m_n); //-- temp = A*x
    LinAlgCUDA::add_mul(m_rhs.val, temp, T(-1), r, m_n); //-- r = b - A*x

    *nbIter=0;

    LinAlgCUDA::norm(r,dot,m_n);
    CUDA_SAFE_CALL(cudaMemcpy(&rho_old, dot, 1 * sizeof(T), cudaMemcpyDeviceToHost));

    if (rho_old/rhs < stop_test) return;

    CUDA_SAFE_CALL(cudaMemcpy(w,r,m_n*sizeof(T),cudaMemcpyDeviceToDevice)); //-- w = r

    // Loop
    while((*nbIter)<nbIterMax){

	LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,w,temp,m_n); //-- temp = A*w

	LinAlgCUDA::dot(w,temp,dot,m_n);
	CUDA_SAFE_CALL(cudaMemcpy(&rho_denum, dot, 1 * sizeof(T), cudaMemcpyDeviceToHost));
	rho=rho_old/rho_denum;

	LinAlgCUDA::add_mul(m_x.val, w, rho, m_x.val, m_n); //-- x = x + rho*w
	LinAlgCUDA::add_mul(r, temp, -rho, r, m_n); //-- r = r - rho*A*w

	LinAlgCUDA::norm(r,dot,m_n);
	CUDA_SAFE_CALL(cudaMemcpy(&rho_new, dot, 1 * sizeof(T), cudaMemcpyDeviceToHost));

	// Stop test
	if (rho_new/rhs < stop_test) break;

	LinAlgCUDA::add_mul(r, w, rho_new/rho_old, w, m_n);

	rho_old=rho_new;

	(*nbIter)++;
    }

    // Cleanup
    CUDA_SAFE_CALL(cudaFree(w));
    CUDA_SAFE_CALL(cudaFree(r));
    CUDA_SAFE_CALL(cudaFree(temp));
    CUDA_SAFE_CALL(cudaFree(norm));
    CUDA_SAFE_CALL(cudaFree(dot));

    ///////////////////////////////
    // Copy solution to CPU
    ///////////////////////////////

    CUDA_SAFE_CALL(cudaMemcpy(res, m_x.val, m_n * sizeof(T), cudaMemcpyDeviceToHost));
}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovGPU<T>::SolveORTH_CSR(T* res, int* nbIter, T tol, int* nbRestart)
{
    int nbIterMax=(*nbIter);
    int nbRestartMax=(*nbRestart);

    ///////////////////////////////
    // Initialization
    ///////////////////////////////

    //LinAlgCUDA::init(m_x.val,T(0),m_n);
    CUDA_SAFE_CALL(cudaMemcpy(m_x.val,res, m_n * sizeof(T), cudaMemcpyHostToDevice));

    ///////////////////////////////
    // ORTHODIR
    ///////////////////////////////

    // Initialization
    int k;
    bool converge=false;
    T rho, rho_num, test, stop_test, alpha_num, rhs, dotAw;
    stop_test=tol*tol;
    T *r, *temp, *norm, *dot, *alpha, *null; null=NULL;
    T **w, **Aw;
    CUDA_SAFE_CALL(cudaMalloc((void**)&r, m_n * sizeof(T)));
    w=(T**)malloc(nbRestartMax * sizeof(T*));
    for(int i=0;i<nbRestartMax;++i)
	CUDA_SAFE_CALL(cudaMalloc((void**)&w[i], m_n * sizeof(T)));
    alpha=(T*)malloc(nbRestartMax * sizeof(T));
    Aw=(T**)malloc(nbRestartMax * sizeof(T*));
    for(int i=0;i<nbRestartMax;++i)
	CUDA_SAFE_CALL(cudaMalloc((void**)&Aw[i], m_n * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&temp, m_n * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&norm, 1 * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dot, 1 * sizeof(T)));
    *nbIter=0;
    *nbRestart=-1;

    LinAlgCUDA::norm(m_rhs.val,norm,m_n);
    CUDA_SAFE_CALL(cudaMemcpy(&rhs, norm, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    if(rhs==0) rhs=1;
    if(rhs<stop_test) rhs=stop_test;

    while(!converge && (*nbIter)<nbIterMax)
	{
	    (*nbRestart)++;

	    // First step
	    LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,m_x.val,temp,m_n); //-- temp = A*x
	    LinAlgCUDA::add_mul(temp,m_rhs.val, T(-1), r, m_n); //-- r = A*x - b

	    LinAlgCUDA::norm(r,norm,m_n);
	    CUDA_SAFE_CALL(cudaMemcpy(&test, norm, 1 * sizeof(T), cudaMemcpyDeviceToHost));
	    if (test/rhs < stop_test) break;

	    CUDA_SAFE_CALL(cudaMemcpy(w[0],r,m_n*sizeof(T),cudaMemcpyDeviceToDevice)); //-- w = r
	    LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,w[0],Aw[0],m_n); //-- Aw = A*w

	    LinAlgCUDA::norm(Aw[0],norm,m_n);
	    CUDA_SAFE_CALL(cudaMemcpy(&dotAw, norm, 1 * sizeof(T), cudaMemcpyDeviceToHost));
	    dotAw=T(1.)/sqrt(dotAw);
	    LinAlgCUDA::add_mul(null,Aw[0], dotAw, Aw[0], m_n); //-- Aw /= ||Aw||
	    LinAlgCUDA::add_mul(null,w[0], dotAw, w[0], m_n);   //-- w  /= ||Aw||

	    LinAlgCUDA::dot(r,Aw[0],dot,m_n);
	    CUDA_SAFE_CALL(cudaMemcpy(&rho_num, dot, 1 * sizeof(T), cudaMemcpyDeviceToHost));
	    rho=-rho_num; //-- rho

	    LinAlgCUDA::add_mul(m_x.val, w[0], rho, m_x.val, m_n); //-- x = x + rho*w

	    k=0;
	    (*nbIter)++;

	    // Loop
	    while(k<(nbRestartMax-1) && (*nbIter)<nbIterMax)
		{
		    LinAlgCUDA::add_mul(r, Aw[k], rho, r, m_n); //-- r = r + rho*A*w

		    // Stop test
		    LinAlgCUDA::norm(r,norm,m_n);
		    CUDA_SAFE_CALL(cudaMemcpy(&test, norm, 1 * sizeof(T), cudaMemcpyDeviceToHost));
		    if (test/rhs < stop_test)
			{
			    converge=true;
			    break;
			}

		    LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,Aw[k],temp,m_n); //-- temp = A*Aw

		    for(int i=0;i<=k;++i)
			{
			    LinAlgCUDA::dot(temp,Aw[i],dot,m_n);
			    CUDA_SAFE_CALL(cudaMemcpy(&alpha_num,dot, 1 * sizeof(T), cudaMemcpyDeviceToHost));
			    alpha[i]=-alpha_num;
			}

		    k++;

		    CUDA_SAFE_CALL(cudaMemcpy(w[k],Aw[k-1],m_n*sizeof(T),cudaMemcpyDeviceToDevice)); //-- w[k+1] = Fw[k]
		    CUDA_SAFE_CALL(cudaMemcpy(Aw[k],temp,m_n*sizeof(T),cudaMemcpyDeviceToDevice)); //-- Aw[k+1] = A*Aw[k]

		    for(int i=0;i<k;++i)
			{
			    LinAlgCUDA::add_mul(w[k], w[i], alpha[i], w[k], m_n); //-- w[k+1] += alpha[i]*w[i]
			    LinAlgCUDA::add_mul(Aw[k], Aw[i], alpha[i], Aw[k], m_n); //-- Aw[k+1] += alpha[i]*Aw[i]
			}

		    LinAlgCUDA::norm(Aw[k],norm,m_n);
		    CUDA_SAFE_CALL(cudaMemcpy(&dotAw, norm, 1 * sizeof(T), cudaMemcpyDeviceToHost));
		    dotAw=T(1.)/sqrt(dotAw);
		    LinAlgCUDA::add_mul(null,Aw[k], dotAw, Aw[k], m_n); //-- Aw /= ||Aw||
		    LinAlgCUDA::add_mul(null,w[k], dotAw, w[k], m_n);   //-- w  /= ||Aw||

		    LinAlgCUDA::dot(r,Aw[k],dot,m_n);
		    CUDA_SAFE_CALL(cudaMemcpy(&rho_num, dot, 1 * sizeof(T), cudaMemcpyDeviceToHost));
		    rho=-rho_num;  //-- rho

		    LinAlgCUDA::add_mul(m_x.val, w[k], rho, m_x.val, m_n); //-- x = x + rho*w

		    (*nbIter)++;
		}
	}

    // Cleanup
    for(int i=0;i<nbRestartMax;++i)
	{
	    CUDA_SAFE_CALL(cudaFree(w[i]));
	    CUDA_SAFE_CALL(cudaFree(Aw[i]));
	}

    CUDA_SAFE_CALL(cudaFree(r));
    CUDA_SAFE_CALL(cudaFree(temp));
    CUDA_SAFE_CALL(cudaFree(norm));
    CUDA_SAFE_CALL(cudaFree(dot));

    free(alpha);
    free(Aw);
    free(w);

    ///////////////////////////////
    // Copy solution to CPU
    ///////////////////////////////

    CUDA_SAFE_CALL(cudaMemcpy(res, m_x.val, m_n * sizeof(T), cudaMemcpyDeviceToHost));

}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovGPU<T>::SolveORTHold_CSR(T* res, int* nbIter, T tol, int* nbRestart)
{
    int nbIterMax=(*nbIter);
    int nbRestartMax=(*nbRestart);

    ///////////////////////////////
    // Initialization
    ///////////////////////////////

    //LinAlgCUDA::init(m_x.val,T(0),m_n);
    CUDA_SAFE_CALL(cudaMemcpy(m_x.val,res, m_n * sizeof(T), cudaMemcpyHostToDevice));

    ///////////////////////////////
    // ORTHODIR
    ///////////////////////////////

    // Initialization
    int k;
    bool converge=false;
    T rho, rho_num, rho_denum, test, stop_test, alpha_num, rhs;
    stop_test=tol*tol;
    T *r, *temp, *norm, *dot, *alpha, *dotAw;
    T **w, **Aw;
    CUDA_SAFE_CALL(cudaMalloc((void**)&r, m_n * sizeof(T)));
    w=(T**)malloc(nbRestartMax * sizeof(T*));
    for(int i=0;i<nbRestartMax;++i)
	CUDA_SAFE_CALL(cudaMalloc((void**)&w[i], m_n * sizeof(T)));
    alpha=(T*)malloc(nbRestartMax * sizeof(T));
    Aw=(T**)malloc(nbRestartMax * sizeof(T*));
    for(int i=0;i<nbRestartMax;++i)
	CUDA_SAFE_CALL(cudaMalloc((void**)&Aw[i], m_n * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&temp, m_n * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&norm, 1 * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&dot, 2 * sizeof(T)));
    dotAw=(T*)malloc(nbRestartMax * sizeof(T));
    *nbIter=0;
    *nbRestart=-1;

    LinAlgCUDA::norm(m_rhs.val,dot,m_n);
    CUDA_SAFE_CALL(cudaMemcpy(&rhs, dot, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    if(rhs==0) rhs=1;
    if(rhs<stop_test) rhs=stop_test;

    while(!converge && (*nbIter)<nbIterMax)
	{
	    (*nbRestart)++;

	    // First step
	    LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,m_x.val,temp,m_n); //-- temp = A*x
	    LinAlgCUDA::add_mul(temp,m_rhs.val, T(-1), r, m_n); //-- r = A*x - b

	    LinAlgCUDA::norm(r,norm,m_n);
	    CUDA_SAFE_CALL(cudaMemcpy(&test, norm, 1 * sizeof(T), cudaMemcpyDeviceToHost));
	    if (test/rhs < stop_test) break;

	    CUDA_SAFE_CALL(cudaMemcpy(w[0],r,m_n*sizeof(T),cudaMemcpyDeviceToDevice)); //-- w = r
	    LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,w[0],Aw[0],m_n); //-- Aw = A*w

	    LinAlgCUDA::dot_by2(r,Aw[0],&dot[0],Aw[0],Aw[0],&dot[1],m_n);
	    CUDA_SAFE_CALL(cudaMemcpy(&rho_num, &dot[0], 1 * sizeof(T), cudaMemcpyDeviceToHost));
	    CUDA_SAFE_CALL(cudaMemcpy(&rho_denum, &dot[1], 1 * sizeof(T), cudaMemcpyDeviceToHost));
	    rho=-rho_num/rho_denum;//-- rho = -(r'*w)/(w'*(A*w))
	    dotAw[0]=rho_denum;

	    LinAlgCUDA::add_mul(m_x.val, w[0], rho, m_x.val, m_n); //-- x = x + rho*w

	    k=0;
	    (*nbIter)++;

	    // Loop
	    while(k<(nbRestartMax-1) && (*nbIter)<nbIterMax)
		{
		    LinAlgCUDA::add_mul(r, Aw[k], rho, r, m_n); //-- r = r + rho*A*w

		    // Stop test
		    LinAlgCUDA::norm(r,norm,m_n);
		    CUDA_SAFE_CALL(cudaMemcpy(&test, norm, 1 * sizeof(T), cudaMemcpyDeviceToHost));
		    if (test/rhs < stop_test)
			{
			    converge=true;
			    break;
			}

		    LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,Aw[k],temp,m_n); //-- temp = A*Aw

		    for(int i=0;i<=k;++i)
			{
			    LinAlgCUDA::dot(temp,Aw[i],&dot[0],m_n);
			    CUDA_SAFE_CALL(cudaMemcpy(&alpha_num, &dot[0], 1 * sizeof(T), cudaMemcpyDeviceToHost));
			    alpha[i]=-alpha_num/dotAw[i];
			}

		    k++;

		    CUDA_SAFE_CALL(cudaMemcpy(w[k],Aw[k-1],m_n*sizeof(T),cudaMemcpyDeviceToDevice)); //-- w[k+1] = Fw[k]
		    CUDA_SAFE_CALL(cudaMemcpy(Aw[k],temp,m_n*sizeof(T),cudaMemcpyDeviceToDevice)); //-- Aw[k+1] = A*Aw[k]

		    for(int i=0;i<k;++i)
			{
			    LinAlgCUDA::add_mul(w[k], w[i], alpha[i], w[k], m_n); //-- w[k+1] += alpha[i]*w[i]
			    LinAlgCUDA::add_mul(Aw[k], Aw[i], alpha[i], Aw[k], m_n); //-- Aw[k+1] += alpha[i]*Aw[i]
			}

		    LinAlgCUDA::dot_by2(r,Aw[k],&dot[0],Aw[k],Aw[k],&dot[1],m_n);
		    CUDA_SAFE_CALL(cudaMemcpy(&rho_num, &dot[0], 1 * sizeof(T), cudaMemcpyDeviceToHost));
		    CUDA_SAFE_CALL(cudaMemcpy(&rho_denum, &dot[1], 1 * sizeof(T), cudaMemcpyDeviceToHost));
		    rho=-rho_num/rho_denum;//-- rho = -(r'*w)/(w'*(A*w))
		    dotAw[k]=rho_denum;

		    LinAlgCUDA::add_mul(m_x.val, w[k], rho, m_x.val, m_n); //-- x = x + rho*w

		    (*nbIter)++;
		}
	}

    // Cleanup
    for(int i=0;i<nbRestartMax;++i)
	{
	    CUDA_SAFE_CALL(cudaFree(w[i]));
	    CUDA_SAFE_CALL(cudaFree(Aw[i]));
	}

    CUDA_SAFE_CALL(cudaFree(r));
    CUDA_SAFE_CALL(cudaFree(temp));
    CUDA_SAFE_CALL(cudaFree(norm));
    CUDA_SAFE_CALL(cudaFree(dot));

    free(alpha);
    free(dotAw);
    free(Aw);
    free(w);

    ///////////////////////////////
    // Copy solution to CPU
    ///////////////////////////////

    CUDA_SAFE_CALL(cudaMemcpy(res, m_x.val, m_n * sizeof(T), cudaMemcpyDeviceToHost));

}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovGPU<T>::SetRHS(T* rhs)
{
    CUDA_SAFE_CALL(cudaMemcpy(m_rhs.val, rhs, m_n * sizeof(T), cudaMemcpyHostToDevice));
}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovGPU<T>::TestMatVec(int compt,T* min_time)
{
    /*
      T *temp,timeMin,timeMean=0;
      CUDA_SAFE_CALL(cudaMalloc((void**)&temp, m_n * sizeof(T)));

      for(int i=0;i<compt;++i){
      TimerGPU time;
      time.StartTimer();
      LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,m_x.val,temp,m_n); //-- test = A*x
      time.StopTimer();
      if(i==0) timeMin=time.GetTimer();
      else{
      if(time.GetTimer()<timeMin)
      timeMin=time.GetTimer();
      }
      timeMean+=time.GetTimer();
      time.~TimerGPU();
      }
      printf("MatVec GPU %f %f\n",timeMin,timeMean/compt);
      CUDA_SAFE_CALL(cudaFree(temp));

      *min_time=timeMin;
      */
    T *temp;
    CUDA_SAFE_CALL(cudaMalloc((void**)&temp, m_n * sizeof(T)));

    TimerGPU time;
    time.StartTimer();

    for(int i=0;i<compt;++i)
	{
	    LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,m_x.val,temp,m_n); //-- test = A*x
	}

    time.StopTimer();
    CUDA_SAFE_CALL(cudaFree(temp));

    *min_time=time.GetTimer()/compt;
}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovGPU<T>::TestDot(int compt,T* min_time)
{
    /*
      T *dot,timeMin,timeMean=0;
      CUDA_SAFE_CALL(cudaMalloc((void**)&dot, 1 * sizeof(T)));

      for(int i=0;i<compt;++i){
      TimerGPU time;
      time.StartTimer();
      LinAlgCUDA::dot(m_rhs.val,m_rhs.val,dot,m_n);
      time.StopTimer();
      if(i==0) timeMin=time.GetTimer();
      else{
      if(time.GetTimer()<timeMin)
      timeMin=time.GetTimer();
      }
      timeMean+=time.GetTimer();
      time.~TimerGPU();
      }
      printf("Dot GPU %f %f\n",timeMin,timeMean/compt);
      CUDA_SAFE_CALL(cudaFree(dot));

      *min_time=timeMin;
      */
    T *dot;
    CUDA_SAFE_CALL(cudaMalloc((void**)&dot, 1 * sizeof(T)));

    TimerGPU time;
    time.StartTimer();

    for(int i=0;i<compt;++i)
	{
	    LinAlgCUDA::dot(m_rhs.val,m_rhs.val,dot,m_n);
	}

    time.StopTimer();

    CUDA_SAFE_CALL(cudaFree(dot));

    *min_time=time.GetTimer()/compt;
}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovGPU<T>::Check(T* abs, T* rel, T* test)
{
    T *norm,*temp;
    T temp_norm;

    CUDA_SAFE_CALL(cudaMalloc((void**)&norm, 1 * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&temp, m_n * sizeof(T)));

    LinAlgCUDA::mat_vec_csr(m_mat.row_index,m_mat.col,m_mat.val,m_x.val,temp,m_n); //-- test = A*x
    LinAlgCUDA::add_mul(temp, m_rhs.val, T(-1), temp, m_n); //-- test = b - A*x

    LinAlgCUDA::norm(temp,norm,m_n);
    CUDA_SAFE_CALL(cudaMemcpy(abs, norm, 1 * sizeof(T), cudaMemcpyDeviceToHost));
    LinAlgCUDA::norm(m_rhs.val,norm,m_n);
    CUDA_SAFE_CALL(cudaMemcpy(&temp_norm, norm, 1 * sizeof(T), cudaMemcpyDeviceToHost));

    if(temp_norm<T(CHECK_EPS))
	{
	    temp_norm=1;
	    printf("relative norm too small (%e): absolute norm instead\n",CHECK_EPS);
	}

    *rel=(*abs)/temp_norm;

    if(test!=NULL)
	{
	    CUDA_SAFE_CALL(cudaMemcpy(test, temp, m_n * sizeof(T), cudaMemcpyDeviceToHost));
	}

    CUDA_SAFE_CALL(cudaFree(temp));
    CUDA_SAFE_CALL(cudaFree(norm));
}
///////////////////////////////////////////////////////////////////////////////////////

template class KrylovGPU<float>;
template class KrylovGPU<double>;

template class KrylovGPU< singlecomplex >;
template class KrylovGPU< doublecomplex >;
