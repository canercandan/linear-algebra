#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil.h>

#include "TimerGPU.h"

TimerGPU::TimerGPU()
{
    //m_timer=0;
    CUT_SAFE_CALL(cutCreateTimer(&m_timer));
}

TimerGPU::~TimerGPU()
{
    CUT_SAFE_CALL(cutDeleteTimer(m_timer));
}

void TimerGPU::StartTimer()
{
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_SAFE_CALL(cutStartTimer(m_timer));
}

void TimerGPU::StopTimer()
{
    CUDA_SAFE_CALL(cudaThreadSynchronize());
    CUT_SAFE_CALL(cutStopTimer(m_timer));
}

float TimerGPU::GetTimer()
{
    return cutGetTimerValue(m_timer);
}
