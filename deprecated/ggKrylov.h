#ifndef ggKrylov_h
#define ggKrylov_h

template < typename GGT >
class ggKrylov
{
public:
    virtual void SolveCG_CSR(GGT* res, int* nbIter, GGT tol){SolveCGv1_CSR(res,nbIter,tol);};

    virtual void SolveCGv1_CSR(GGT* res, int* nbIter, GGT tol) = 0;
    virtual void SolveCGv2_CSR(GGT* res, int* nbIter, GGT tol) = 0;
    virtual void SolveORTH_CSR(GGT* res, int* nbIter, GGT tol, int* nbRestart) = 0;
    virtual void SetRHS(GGT* rhs) = 0;
    virtual void Check(GGT* abs, GGT* rel, GGT* test=NULL) = 0;
    virtual void TestMatVec(int compt,GGT* min_time) = 0;
    virtual void TestDot(int compt,GGT* min_time) = 0;
};

#endif // !ggKrylov_h
