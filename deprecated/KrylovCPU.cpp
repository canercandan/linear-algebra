#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil.h>

#include "KrylovCPU.hpp"

#include "cudacomplex.h"

template <class T>
KrylovCPU<T>::KrylovCPU(int n, T* val_rhs, int nnz, int* row_index, int* col, T* val_mat)
{
    ///////////////////////////////
    // Data initialization on GPU
    ///////////////////////////////

    m_n=n;
    m_mat.n=n;
    m_mat.nnz=nnz;
    m_rhs.n=n;
    m_x.n=n;
    m_mat.row_index=new int[n+1];
    m_mat.col=new int[nnz];
    m_mat.val=new T[nnz];
    m_rhs.val=new T[n];
    m_x.val=new T[n];

    ///////////////////////////////
    // Data copy to CPU
    ///////////////////////////////

    memcpy(m_mat.row_index, row_index, (n+1) * sizeof(int));
    memcpy(m_mat.col, col, nnz * sizeof(int));
    memcpy(m_mat.val, val_mat, nnz * sizeof(T));
    memcpy(m_rhs.val, val_rhs, n * sizeof(T));

    memset(m_x.val,0,m_n*sizeof(T));
}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
KrylovCPU<T>::~KrylovCPU(){

    ///////////////////////////////
    // Cleanup
    ///////////////////////////////

    delete[] (m_mat.row_index);
    delete[] (m_mat.col);
    delete[] (m_mat.val);
    delete[] (m_x.val);
    delete[] (m_rhs.val);
}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovCPU<T>::SolveCGv1_CSR(T* res, int* nbIter, T tol){

    int nbIterMax=(*nbIter);

    ///////////////////////////////
    // Initialization
    ///////////////////////////////

    memcpy(m_x.val,res,m_n*sizeof(T));

    ///////////////////////////////
    // Conjugate Gradient
    ///////////////////////////////

    // Initialization
    T rho, rho_num, rho_denum, stop_test, test, rhs;
    stop_test=tol*tol;
    T *r, *w, *temp;
    r = new T[m_n];
    w = new T[m_n];
    temp = new T[m_n];

    // Stop test
    norm(m_rhs.val,&rhs,m_n);
    if (rhs == 0 ) rhs=1;
    if(rhs<stop_test) rhs=stop_test;

    // First step
    mat_vec_csr(&m_mat,m_x.val,temp); //-- temp = A*x
    add_mul(m_rhs.val, temp, T(-1), r, m_n); //-- r = b - A*x

    norm(r,&test,m_n);
    if (test/rhs < stop_test) return;

    memcpy(w,r,m_n*sizeof(T)); //-- w = r
    mat_vec_csr(&m_mat,w,temp); //-- temp = A*w

    dot(r,w,&rho_num,m_n);
    dot(w,temp,&rho_denum,m_n);
    rho=rho_num/rho_denum;//-- rho = (r'*w)/(w'*(A*w))

    add_mul(m_x.val, w, rho, m_x.val, m_n); //-- x = x + rho*w

    // Loop
    *nbIter=1;
    while((*nbIter)<nbIterMax)
	{
	    add_mul(r, temp, -rho, r, m_n); //-- r = r - rho*A*w

	    // Stop test
	    norm(r,&test,m_n);
	    if (test/rhs < stop_test) break;

	    dot(r,temp,&rho_num,m_n);
	    dot(w,temp,&rho_denum,m_n);
	    rho=rho_num/rho_denum;//-- rho = (r'*A*w)/(w'*(A*w))
	    add_mul(r, w, -rho, w, m_n); //-- w = r - rho*w

	    mat_vec_csr(&m_mat,w,temp); //-- temp = A*w
	    dot(r,w,&rho_num,m_n);
	    dot(w,temp,&rho_denum,m_n);
	    rho=rho_num/rho_denum;//-- rho = (r'*w)/(w'*(A*w))
	    add_mul(m_x.val, w, rho, m_x.val, m_n); //-- x = x + rho*w

	    (*nbIter)++;
	}
    // Cleanup
    delete[] w;
    delete[] r;
    delete[] temp;

    ///////////////////////////////
    // Copy solution to CPU
    ///////////////////////////////

    memcpy(res, m_x.val, m_n * sizeof(T));

}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovCPU<T>::SolveCGv2_CSR(T* res, int* nbIter, T tol){

    int nbIterMax=(*nbIter);

    ///////////////////////////////
    // Initialization
    ///////////////////////////////

    memset(m_x.val,0,m_n*sizeof(T));

    ///////////////////////////////
    // Conjugate Gradient
    ///////////////////////////////

    // Initialization
    T rho, rho_new, rho_old, rho_denum, stop_test, rhs;
    stop_test=tol*tol;
    T *r, *w, *temp;
    r = new T[m_n];
    w = new T[m_n];
    temp = new T[m_n];

    // Stop test
    norm(m_rhs.val,&rhs,m_n);
    if (rhs == 0 ) rhs=1;
    if(rhs<stop_test) rhs=stop_test;

    // First step
    mat_vec_csr(&m_mat,m_x.val,temp); //-- temp = A*x
    add_mul(m_rhs.val, temp, T(-1), r, m_n); //-- r = b - A*x

    *nbIter=0;

    norm(r,&rho_old,m_n);
    if (rho_old/rhs < stop_test) return;

    memcpy(w,r,m_n*sizeof(T)); //-- w = r

    // Loop
    while((*nbIter)<nbIterMax)
	{
	    mat_vec_csr(&m_mat,m_x.val,temp); //-- temp = A*x

	    dot(w,temp,&rho_denum,m_n);
	    rho=rho_old/rho_denum;

	    add_mul(m_x.val, w, rho, m_x.val, m_n); //-- x = x + rho*w
	    add_mul(r, temp, -rho, r, m_n); //-- r = r - rho*A*w

	    norm(r,&rho_new,m_n);

	    // Stop test
	    if (rho_new/rhs < stop_test) break;

	    add_mul(r, w, rho_new/rho_old, w, m_n);

	    rho_old=rho_new;

	    (*nbIter)++;
	}

    // Cleanup
    delete[] w;
    delete[] r;
    delete[] temp;

    ///////////////////////////////
    // Copy solution to CPU
    ///////////////////////////////

    memcpy(res, m_x.val, m_n * sizeof(T));

}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovCPU<T>::SolveORTH_CSR(T* res, int* nbIter, T tol, int *nbRestart){

    int nbRestartMax=(*nbRestart);
    int nbIterMax=(*nbIter);

    ///////////////////////////////
    // Initialization
    ///////////////////////////////

    memset(m_x.val,0,m_n*sizeof(T));

    ///////////////////////////////
    // ORTHODIR
    ///////////////////////////////

    // Initialization
    int k;
    bool converge=false;
    T rho, rho_num, rho_denum, test, stop_test, alpha_num;
    stop_test=tol*tol;
    T *r, *temp, *alpha, *dotAw;
    T **w, **Aw;
    r=new T[m_n];
    w=new T*[nbRestartMax];
    for(int i=0;i<nbRestartMax;++i)
	w[i]=new T[m_n];
    alpha=new T[nbRestartMax];
    Aw=new T*[nbRestartMax];
    for(int i=0;i<nbRestartMax;++i)
	Aw[i]=new T[m_n];
    temp=new T[m_n];
    dotAw=new T[nbRestartMax];
    *nbIter=0;
    *nbRestart=-1;

    while(!converge && (*nbIter)<nbIterMax)
	{
	    (*nbRestart)++;

	    // First step
	    mat_vec_csr(&m_mat,m_x.val,temp); //-- temp = A*x
	    add_mul(temp,m_rhs.val, T(-1), r, m_n); //-- r = A*x - b
	    memcpy(w[0],r,m_n*sizeof(T)); //-- w = r
	    mat_vec_csr(&m_mat,w[0],Aw[0]); //-- Aw = A*w

	    dot(r,Aw[0],&rho_num,m_n);
	    dot(Aw[0],Aw[0],&rho_denum,m_n);
	    rho=-rho_num/rho_denum;//-- rho = -(r'*w)/(w'*(A*w))
	    dotAw[0]=rho_denum;

	    add_mul(m_x.val, w[0], rho, m_x.val, m_n); //-- x = x + rho*w

	    k=0;
	    (*nbIter)++;
	    // Loop
	    while(k<(nbRestartMax-1) && (*nbIter)<nbIterMax){
		add_mul(r, Aw[k], rho, r, m_n); //-- r = r + rho*A*w

		// Stop test
		norm(r,&test,m_n);
		if (test < stop_test)
		    {

			converge=true;
			break;
		    }

		mat_vec_csr(&m_mat,Aw[k],temp); //-- temp = A*Aw

		for(int i=0;i<=k;++i){
		    dot(temp,Aw[i],&alpha_num,m_n);
		    alpha[i]=-alpha_num/dotAw[i];
		}

		k++;

		memcpy(w[k],Aw[k-1],m_n*sizeof(T)); //-- w[k+1] = Fw[k]
		memcpy(Aw[k],temp,m_n*sizeof(T));  //-- Aw[k+1] = A*Aw[k]
		for(int i=0;i<k;++i)
		    {
			add_mul(w[k], w[i], alpha[i], w[k], m_n); //-- w[k+1] += alpha[i]*w[i]
			add_mul(Aw[k], Aw[i], alpha[i], Aw[k], m_n); //-- Aw[k+1] += alpha[i]*Aw[i]
		    }

		dot(r,Aw[k],&rho_num,m_n);
		dot(Aw[k],Aw[k],&rho_denum,m_n);
		rho=-rho_num/rho_denum;//-- rho = -(r'*w)/(w'*(A*w))
		dotAw[k]=rho_denum;

		add_mul(m_x.val, w[k], rho, m_x.val, m_n); //-- x = x + rho*w

		(*nbIter)++;
	    }
	}

    // Cleanup
    for(int i=0;i<nbRestartMax;++i)
	{
	    delete[] w[i];
	    delete[] Aw[i];
	}

    delete[] w;
    delete[] Aw;
    delete[] r;
    delete[] temp;
    delete[] alpha;
    delete[] dotAw;

    ///////////////////////////////
    // Copy solution to CPU
    ///////////////////////////////

    memcpy(res,m_x.val,m_n*sizeof(T));
}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovCPU<T>::SetRHS(T* rhs)
{
    memcpy(m_rhs.val, rhs, m_n * sizeof(T));
}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovCPU<T>::TestMatVec(int compt,T* min_time)
{
    /*
      T *temp,timeMin,timeMean=0;
      temp = new T[m_n];

      for(int i=0;i<compt;++i){
      TimerGPU time;
      time.StartTimer();
      mat_vec_csr(&m_mat,m_x.val,temp); //-- test = A*x
      time.StopTimer();
      if(i==0) timeMin=time.GetTimer();
      else{
      if(time.GetTimer()<timeMin)
      timeMin=time.GetTimer();
      }
      timeMean+=time.GetTimer();
      time.~TimerGPU();
      }
      printf("MAtvecCPU %f %f\n",timeMin,timeMean/compt);
      delete[] temp;

      *min_time=timeMin;
      */
    T *temp;
    temp = new T[m_n];

    TimerGPU time;
    time.StartTimer();

    for(int i=0;i<compt;++i)
	{
	    mat_vec_csr(&m_mat,m_x.val,temp); //-- test = A*x
	}

    time.StopTimer();

    delete[] temp;

    *min_time=time.GetTimer()/compt;
}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovCPU<T>::TestDot(int compt,T* min_time)
{
    /*
      T dott,timeMin,timeMean=0;

      for(int i=0;i<compt;++i){
      TimerGPU time;
      time.StartTimer();
      dot(m_rhs.val,m_rhs.val,&dott,m_n);
      time.StopTimer();
      if(i==0) timeMin=time.GetTimer();
      else{
      if(time.GetTimer()<timeMin)
      timeMin=time.GetTimer();
      }
      timeMean+=time.GetTimer();
      time.~TimerGPU();
      }
      printf("Dot CPU %f %f\n",timeMin,timeMean/compt);
      *min_time=timeMin;
      */
    T dott;

    TimerGPU time;
    time.StartTimer();

    for(int i=0;i<compt;++i)
	{
	    dot(m_rhs.val,m_rhs.val,&dott,m_n);
	}

    time.StopTimer();

    *min_time=time.GetTimer()/compt;
}
///////////////////////////////////////////////////////////////////////////////////////
template <class T>
void KrylovCPU<T>::Check(T* abs, T* rel, T* test)
{
    T *temp;
    T res_norm;

    temp = new T[m_n];

    mat_vec_csr(&m_mat,m_x.val,temp); //-- test = A*x
    add_mul(temp, m_rhs.val, T(-1), temp, m_n); //-- test = b - A*x

    norm(temp,abs,m_n);
    norm(m_rhs.val,&res_norm,m_n);

    if(res_norm<T(CHECK_EPS))
	{
	    res_norm=1;
	    printf("relative norm too small (%e): absolute norm instead\n",CHECK_EPS);
	}

    *rel=(*abs)/res_norm;

    if(test!=NULL)
	{
	    memcpy(test, temp, m_n * sizeof(T));
	}

    delete[] temp;
}
///////////////////////////////////////////////////////////////////////////////////////

template class KrylovCPU<float>;
template class KrylovCPU<double>;

template class KrylovCPU< singlecomplex >;
template class KrylovCPU< doublecomplex >;
