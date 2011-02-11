#ifndef TimerGPU_h
#define TimerGPU_h

class TimerGPU
{
private:
    unsigned int m_timer;
public:
    TimerGPU();
    ~TimerGPU();
public:
    void StartTimer();
    void StopTimer();
    float GetTimer();
};

#endif // !TimerGPU_h
