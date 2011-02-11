#ifndef _ggMarging_h
#define _ggMarging_h

#include "ggObject.h"

class ggMarging : public ggObject
{
public:
    virtual void pre() = 0;
    virtual void post() = 0;
};

#endif // !_ggMarging_h
