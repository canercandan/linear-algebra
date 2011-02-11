// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

//-----------------------------------------------------------------------------
// ggTimerCuda.h
/*
  Contact: Caner Candan <caner@candan.fr>
*/
//-----------------------------------------------------------------------------

#ifndef _ggTimerCuda_h
#define _ggTimerCuda_h

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil.h>

#include "ggPrintable.h"

class ggTimerCuda : public ggMarging, public ggPrintable
{
public:
    ggTimerCuda() { CUT_SAFE_CALL(cutCreateTimer(&m_timer)); }
    ~ggTimerCuda() { CUT_SAFE_CALL(cutDeleteTimer(m_timer)); }

    void pre()
    {
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStartTimer(m_timer));
    }

    void post()
    {
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(m_timer));
    }

    void printOn( std::ostream& os ) const
    {
	os << "Elapsed time: " << _timer << std::endl;
    }

private:
    unsigned int _timer;
};

#endif // !_ggTimerCuda_h
